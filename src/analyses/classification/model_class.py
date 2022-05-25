import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.metrics import average_precision_score
import training.ln_lstm as lstm
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class MultiClass(nn.Module):
    """
    Hi-C-LSTM model.
    Includes Embedding layer for positional representations.
    Layer-Norm LSTM for as the decoder.
    """

    def __init__(self, cfg, device):
        super(MultiClass, self).__init__()
        self.device = device
        self.cfg = cfg
        self.hidden_size_lstm = cfg.hidden_size_lstm
        self.gpu_id = 0
        self.model_name = cfg.model_name
        self.df_columns = [str(i) for i in range(0, 16)]
        self.class_columns = [str(i) for i in range(0, 10)]

        "criterion"
        self.multi_label_criterion = torch.nn.BCEWithLogitsLoss()

        "Initializes linear layers"
        self.multi_label = nn.Linear(cfg.pos_embed_size, 10)

    def forward(self, embed, mode):
        """
        forward_reinit(self, input) -> tensor
        Default forward method that reinitializes hidden sates in every frame.
        Args:
            input (Tensor): The concatenated pairwise indices.
        """

        pred = self.multi_label(embed)

        if mode == "test":
            pred = self.sigm(pred)

        return pred

    def compile_optimizer(self):
        """
        compile_optimizer(self) -> optimizer, criterion
        Method to initialize optimizer and criterion
        Args:
            No Args, specify learning rate in config. Uses Adam as default.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)

        return optimizer

    def load_weights(self):
        """
        load_weights(self) -> No return object
        Method to load model weights. Loads state dict into model object directly
        Args:
            No Args, specify model name and directory in config.
        Raises:
            Error: If exception when loading weights. Eg. if weights dont exist
        """
        try:
            print('loading weights from {}'.format(self.cfg.class_model_dir))
            self.load_state_dict(torch.load(self.cfg.class_model_dir + self.class_model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is None:
                    ave_grads.append(0)
                    max_grads.append(0)
                else:
                    ave_grads.append(p.grad.cpu().abs().mean())
                    max_grads.append(p.grad.cpu().abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()
        print("done")

    def train_model_multi(self, embed_model, epoch, optimizer, writer, feature_matrix, iter, cfg):
        """
        train_model(self, data_loader, criterion, optimizer, writer) -> No return object
        Method to train the model.
        Runs the indices through the forward method. Gets output Hi-C.
        Compares with observed Hi-C. Computes Criterion. Saves trained model.
        Args:
            optimizer (Optimizer): Adam like with learning rate.
            writer (SummaryWriter): Tensorboard SummaryWriter
        """
        device = self.device
        batch_size = cfg.batch_size
        batches = int(np.ceil(len(feature_matrix) / batch_size))

        with torch.autograd.set_detect_anomaly(True):
            self.train()
            running_loss = 0.0
            for i in range(batches):
                batch_pos = torch.tensor(
                    feature_matrix.iloc[i * batch_size:(i + 1) * batch_size]["pos"].values.astype(int)).to(device)
                batch_target = torch.tensor(
                    feature_matrix.iloc[i * batch_size:(i + 1) * batch_size][self.class_columns].values).float().to(
                    device)

                if i == batches - 1:
                    batch_pos = torch.tensor(
                        feature_matrix.iloc[i * batch_size:]["pos"].values.astype(int)).to(device)
                    batch_target = torch.tensor(
                        feature_matrix.iloc[i * batch_size:][self.class_columns].values).float().to(device)

                batch_embed = embed_model.pos_embed(batch_pos.long())

                "Forward Pass"
                batch_pred = self.forward(batch_embed, "train")
                loss = self.multi_label_criterion(batch_pred, batch_target)

                "Backward and optimize"
                optimizer.zero_grad()
                loss.backward()
                # self.plot_grad_flow(self.named_parameters())
                clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                optimizer.step()

                running_loss += loss.item()
                writer.add_scalar('classification testing loss', loss, iter + i)

                "save model"
                torch.save(self.state_dict(), cfg.model_dir + self.model_name + '.pth')

            epoch_loss = running_loss / batches
            print('epoch %s - loss : %s' % (epoch, epoch_loss))

        return iter + i, self, epoch_loss

    def test_model_multi(self, feature_matrix, cfg, embed_model):
        """
        test(self, data_loader, embed_rows) -> tensor, tensor, tensor, DataFrame, Array
        Method to test the given model. Computes error after forward pass.
        Runs post processing to compute error and get embeddings.
        Return
        Args:
            data_loader (DataLoader): DataLoader containing dataset.
            embed_rows (Array): Array of embeddings
        """

        device = self.device
        batch_size = cfg.batch_size
        batches = int(np.ceil(len(feature_matrix) / batch_size))

        with torch.no_grad():
            self.eval()
            running_map = 0.0
            for i in range(batches):
                batch_pos = torch.tensor(
                    feature_matrix.iloc[i * batch_size:(i + 1) * batch_size]["pos"].values.astype(int)).to(device)
                batch_target = feature_matrix.iloc[i * batch_size:(i + 1) * batch_size][self.class_columns].astype(int)

                if i == batches - 1:
                    batch_pos = torch.tensor(
                        feature_matrix.iloc[i * batch_size:]["pos"].values.astype(int)).to(device)
                    batch_target = feature_matrix.iloc[i * batch_size:][self.class_columns].astype(int)

                batch_embed = embed_model.pos_embed(batch_pos.long())

                "Forward Pass"
                batch_pred = self.forward(batch_embed, "test")
                batch_pred = batch_pred.cpu().numpy()
                mAP = average_precision_score(batch_target, batch_pred)

                running_map += mAP

        mean_mAP = running_map / batches
        print('mAP : %s' % (mean_mAP))
        return mean_mAP, self
