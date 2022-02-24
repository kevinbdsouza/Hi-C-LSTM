import torch
from torch import nn
from torch.autograd import Variable
import training.ln_lstm as lstm
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.nn.utils.clip_grad import clip_grad_norm_


class Decoder(nn.Module):
    """
    Class includes Decoders to be used with representations.
    Includes LSTM, CNN, and FC decoders.
    """

    def __init__(self, cfg, device, decoder="lstm"):
        super(Decoder, self).__init__()
        self.device = device
        self.cfg = cfg
        self.hidden_size_lstm = cfg.hidden_size_lstm
        self.gpu_id = 0
        self.decoder_name = cfg.decoder_name

        "intialize LSTM, CNN, and FC decoders"
        self.sigm = nn.Sigmoid()
        if decoder == "lstm":
            self.decoder_lstm = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
            self.decoder_lstm_fc = nn.Linear(cfg.hidden_size_lstm * cfg.sequence_length,
                                             cfg.output_size_lstm * cfg.sequence_length)
        elif decoder == "cnn":
            self.decoder_cnn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=(2, 1), padding=1),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=2, stride=2)).to(self.device)

            self.decoder_cnn_fc = nn.Linear((self.cfg.sequence_length // 2 + 1) * self.cfg.hidden_size_lstm,
                                            self.cfg.output_size_lstm * self.cfg.sequence_length).to(self.device)
        elif decoder == "fc":
            self.decoder_fc = nn.Linear(cfg.input_size_lstm * cfg.sequence_length,
                                        cfg.output_size_lstm * cfg.sequence_length)
        else:
            print("Decoder should be one of lstm, cnn, and fc.")

    def lstm_decoder(self, embeddings):
        """
        lstm_decoder(self, embeddings) -> tensor
        Forward method to be used when using LSTM as the decoder with representations.
        Args:
            embeddings (Tensor): The embeddings for positions.
        """
        hidden, state = self._initHidden(embeddings.shape[0])
        output, _ = self.decoder_lstm(embeddings, (hidden, state))
        output = self.decoder_lstm_fc(output.reshape(embeddings.shape[0], -1))
        output = self.sigm(output)
        return output

    def cnn_decoder(self, embeddings):
        """
        cnn_decoder(self, embeddings) -> tensor
        Forward method to be used when using CNN as the decoder with representations.
        Args:
            embeddings (Tensor): The embeddings for positions.
        """
        output = self.decoder_cnn(embeddings)
        output = torch.permute(output, (0, 3, 2, 1)).squeeze(3)
        output = self.decoder_cnn_fc(output.reshape(embeddings.shape[0], -1))
        output = self.sigm(output)
        return output

    def fc_decoder(self, embeddings):
        """
        fc_decoder(self, embeddings) -> tensor
        Forward method to be used when using FC as the decoder with representations.
        Args:
            embeddings (Tensor): The embeddings for positions.
        """
        output = self.decoder_fc(embeddings)
        output = self.sigm(output)
        return output

    def _initHidden(self, batch_size):
        """
        _initHidden(self, batch_size) -> tensor, tensor
        Method to initialize hidden and cell state
        Args:
            batch_size (int): Batch size, usually the first dim of input data
        """
        h = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        c = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)

        return h, c

    def compile_optimizer(self):
        """
        compile_optimizer(self) -> optimizer, criterion
        Method to initialize optimizer and criterion
        Args:
            No Args, specify decoder learning rate in config. Uses Adam as default.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.dec_learning_rate)
        criterion = nn.MSELoss()

        return optimizer, criterion

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
            print('loading decoder weights from {}'.format(self.cfg.decoder_name))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.decoder_name + '.pth'))
        except Exception as e:
            print("load decoder weights exception: {}".format(e))

    def remove_padded_indices(self, ind, val, pred, pred_df):
        """
        remove_padded_indices(ind, val, pred, pred_df) -> DataFrame
        remove padded indices from all arrays.
        Args:
            ind (Array): Concatenated indices
            val (Array): The Hi-C observed values
            pred (Array): The Hi-C predicted values
            pred_df (DataFrame): Dataframe to add knockout values with indices
        """
        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        val = np.delete(val, idx, axis=0)
        pred = np.delete(pred, idx, axis=0)
        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["v"] = val
        pred_df["pred"] = pred
        return pred_df

    def train_decoder(self, data_loader, embed_rows, start, criterion, optimizer, writer, decoder="lstm"):
        """
        train_decoder(data_loader, embed_rows, start, criterion, optimizer, writer, decoder="lstm") -> No return object
        Method to train the decoders.
        Runs the representations from specific method through the chosen decoder. Saves the resulting decoder model.
        Args:
            data_loader (DataLoader): DataLoader containing dataset
            embed_rows (Array): embeddings
            start (int):  shift value
            criterion (Criterion): Loss function
            optimizer (Optimizer): Adam like with learning rate.
            writer (SummaryWriter): Tensorboard SummaryWriter
            decoder (string): one of lstm, cnn, and fc
        """
        device = self.device
        cfg = self.cfg
        num_epochs = cfg.decoder_epochs

        for epoch in range(num_epochs):
            print(epoch)
            with torch.autograd.set_detect_anomaly(True):
                self.train()
                running_loss = 0.0
                for i, (indices, values) in enumerate(tqdm(data_loader)):
                    ind = indices.cpu().detach().numpy().reshape(-1, 2)
                    values = values.to(device)
                    embed_ij = np.zeros((ind.shape[0], 2 * cfg.pos_embed_size))

                    for n in range(ind.shape[0]):
                        "replaces padding with mean"
                        if ind[n, 0] == 0 and ind[n, 1] == 0:
                            embed_ij[n, 0:cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                            embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                        else:
                            embed_ij[n, 0:cfg.pos_embed_size] = embed_rows[int(ind[n, 0]) - start]
                            embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = embed_rows[int(ind[n, 1]) - start]

                    embeddings = torch.from_numpy(embed_ij)

                    "run decoder with representations"
                    if decoder == "lstm":
                        embeddings = embeddings.view((-1, self.cfg.sequence_length, 2 * cfg.pos_embed_size)).float().to(device)
                        output = self.lstm_decoder(embeddings)
                    elif decoder == "cnn":
                        embeddings = embeddings.view(
                            (self.cfg.batch_size, self.cfg.sequence_length, self.cfg.pos_embed_size, -1))
                        embeddings = torch.permute(embeddings, (0, 3, 2, 1))
                        output = self.cnn_decoder(embeddings)
                    elif decoder == "fc":
                        output = self.fc_decoder(embeddings)

                    loss = criterion(output, values)

                    "Backward and optimize"
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                    optimizer.step()

                    running_loss += loss.item()
                    writer.add_scalar('training loss',
                                      loss, i + epoch * len(data_loader))

            "save model"
            torch.save(self.state_dict(), cfg.model_dir + self.decoder_name + '.pth')
            print('Completed Decoder epoch %s' % str(epoch + 1))
            print('Average Decoder loss: %s' % (running_loss / len(data_loader)))

    def test_decoder(self, data_loader, embed_rows, start, decoder="lstm"):
        """
        test_decoder(data_loader, embed_rows, start, decoder) -> tensor, DataFrame
        Method to test the decoder.
        Runs the representations from specific method through chosen decoder. Gets output Hi-C.
        Save resulting representations and predictions in csv file.
        Args:
            data_loader (DataLoader): DataLoader containing dataset
            embed_rows (Array): embeddings
            start (int):  shift value
            decoder (string): one of lstm, cnn, and fc
        """
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length
        ko_predictions = torch.empty(0, num_outputs).to(device)
        main_pred_df = pd.DataFrame(columns=["i", "j", "v", "pred"])

        with torch.no_grad():
            self.eval()
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                pred_df = pd.DataFrame(columns=["i", "j", "v", "pred"])
                ind = indices.cpu().detach().numpy().reshape(-1, 2)
                val = values.cpu().detach().numpy().reshape(-1, 1)
                embed_ij = np.zeros((ind.shape[0], 2 * cfg.pos_embed_size))

                for n in range(ind.shape[0]):
                    "replaces padding with mean"
                    if ind[n, 0] == 0 and ind[n, 1] == 0:
                        embed_ij[n, 0:cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                        embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                    else:
                        embed_ij[n, 0:cfg.pos_embed_size] = embed_rows[int(ind[n, 0]) - start]
                        embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = embed_rows[int(ind[n, 1]) - start]

                embeddings = torch.from_numpy(embed_ij)
                embeddings = embeddings.view((-1, self.cfg.sequence_length, 2 * cfg.pos_embed_size)).float().to(
                    device)

                "run decoder with representations"
                if decoder == "lstm":
                    output = self.lstm_decoder(embeddings)
                elif decoder == "cnn":
                    output = self.cnn_decoder(embeddings)
                elif decoder == "fc":
                    output = self.fc_decoder(embeddings)

                predictions = torch.cat((ko_predictions, output), 0)

                "run postprocessing"
                pred = output.cpu().detach().numpy().reshape(-1, 1)
                pred_df = self.remove_padded_indices(ind, val, pred, pred_df)

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        predictions = torch.reshape(predictions, (-1, 1)).cpu().detach().numpy()
        return predictions, main_pred_df
