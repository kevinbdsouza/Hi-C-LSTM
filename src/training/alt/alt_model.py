import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
import training.ln_lstm as lstm
from training.alt.alt_data_utils import get_data, get_cumpos


class FullMLP(nn.Module):
    """
    Full MLP.
    """

    def __init__(self, cfg, device):
        super(FullMLP, self).__init__()
        self.device = device
        self.cfg = cfg

        "Initializes FC"
        self.fc1 = nn.Linear(cfg.input_size_mlp, cfg.hidden_size_fc2)
        self.fc2 = nn.Linear(cfg.hidden_size_fc2, cfg.output_size_mlp)
        self.sigm = nn.Sigmoid()

        self.criterion = nn.MSELoss()

    def forward(self, input_pair, values, cum_pos, full_reps):
        """
        forward(self, input_pair, values, cum_pos) -> tensor, tensor
        Default forward method for MLP.
        Args:
            input (Tensor): The concatenated pairwise indices.
        """

        input_pair = input_pair - cum_pos
        input_pair[input_pair < 0] = 0

        input_reps = full_reps[input_pair]
        input_reps = input_reps.view((-1, self.cfg.input_size_mlp))
        output_fc = self.fc1(input_reps)
        output_fc = self.fc2(output_fc)
        output_fc = self.sigm(output_fc).squeeze(1)

        rows = input_pair[:, 0]
        columns = input_pair[:, 1]

        value_pairs = values[rows, columns]
        return self.criterion(output_fc, value_pairs)


class SeqLSTM(nn.Module):
    """
    Hi-C-LSTM model.
    Includes Embedding layer for positional representations.
    Layer-Norm LSTM for as the decoder.
    """

    def __init__(self, cfg, device):
        super(SeqLSTM, self).__init__()
        self.device = device
        self.cfg = cfg
        self.gpu_id = 0
        self.model_name = cfg.model_name

        "Initializes ebedding layer"
        self.pos_embed_layer = nn.Embedding(cfg.genome_len, cfg.hs_pos_lstm).train()
        nn.init.normal_(self.pos_embed_layer.weight)

        "Initializes BiLSTMs"
        self.pos_lstm = lstm.LSTM(cfg.hs_pos_lstm, cfg.hs_pos_lstm, bidirectional=1, batch_first=True)
        self.mb_lstm = lstm.LSTM(cfg.hs_pos_lstm, cfg.hs_mb_lstm, bidirectional=1, batch_first=True)
        self.mega_lstm = lstm.LSTM(cfg.hs_mb_lstm, cfg.hs_mega_lstm, bidirectional=1, batch_first=True)

        self.fullMLP = FullMLP(cfg, device)

        "freezes LSTM during training"
        if cfg.lstm_nontrain:
            self.pos_lstm.requires_grad = False
            self.mb_lstm.requires_grad = False
            self.mega_lstm.requires_grad = False
            self.pos_embed_layer.requires_grad = True
            self.fc1.requires_grad = False
            self.fc2.requires_grad = False

    def forward(self, input, nrows):
        """
        forward(self, input, nrows) -> tensor, tensor
        Default forward method that reinitializes hidden sates in every frame.
        Args:
            input (Tensor): The concatenated pairwise indices.
        """

        input = input.view(-1, self.cfg.sequence_length_pos)
        n_mb = input.shape[0]

        pos_reps = self.pos_embed_layer(input.long())
        pos_reps = pos_reps.view((input.shape[0], self.cfg.sequence_length_pos, -1))

        hidden_pos, state_pos = self._initHidden(input.shape[0], self.cfg.hs_pos_lstm)
        output_pos, (hidden_pos, _) = self.pos_lstm(pos_reps, (hidden_pos, state_pos))
        output_pos = output_pos.reshape((-1, self.cfg.hs_pos_lstm, 2))
        output_pos = torch.mean(output_pos, 2)

        hidden_pos = torch.mean(hidden_pos, 0)
        pad_len = self.cfg.sequence_length_mb - (hidden_pos.size()[0] % self.cfg.sequence_length_mb)
        pad = torch.zeros(pad_len, self.cfg.hs_pos_lstm).to(self.device)
        hidden_pos = torch.cat([hidden_pos, pad], 0)
        hidden_pos = hidden_pos.view((-1, self.cfg.sequence_length_mb, self.cfg.hs_pos_lstm))

        hidden_mb, state_mb = self._initHidden(hidden_pos.shape[0], self.cfg.hs_mb_lstm)
        output_mb, (hidden_mb, _) = self.mb_lstm(hidden_pos, (hidden_mb, state_mb))
        output_mb = output_mb.reshape((-1, self.cfg.hs_mb_lstm, 2))
        output_mb = torch.mean(output_mb, 2)

        hidden_mb = torch.mean(hidden_mb, 0)
        n_mega = hidden_mb.size()[0]
        pad_len = self.cfg.sequence_length_mega - (hidden_mb.size()[0] % self.cfg.sequence_length_mega)
        pad = torch.zeros(pad_len, self.cfg.hs_mb_lstm).to(self.device)
        hidden_mb = torch.cat([hidden_mb, pad], 0)
        hidden_mb = hidden_mb.unsqueeze(0)

        hidden_mega, state_mega = self._initHidden(hidden_mb.shape[0], self.cfg.hs_mega_lstm)
        output_mega, (hidden_mega, _) = self.mega_lstm(hidden_mb, (hidden_mega, state_mega))
        output_mega = output_mega.reshape((-1, self.cfg.hs_mega_lstm, 2))
        output_mega = torch.mean(output_mega, 2)

        full_reps = self.combine_reps(output_pos, output_mb, output_mega, n_mega, n_mb, nrows)
        return full_reps

    def _initHidden(self, batch_size, hidden_size):
        """
        _initHidden(self, batch_size) -> tensor, tensor
        Method to initialize hidden and cell state
        Args:
            batch_size (int): Batch size, usually the first dim of input data
        """
        h = Variable(torch.randn(2, batch_size, hidden_size)).to(self.device)
        c = Variable(torch.randn(2, batch_size, hidden_size)).to(self.device)

        return h, c

    def combine_reps(self, output_pos, output_mb, output_mega, n_mega, n_mb, nrows):
        zero_embed = torch.cat([output_pos[-1], output_mb[-1], output_mega[-1]], 0).unsqueeze(0)

        output_mb_fit = output_mb[:n_mb, :]
        output_mega_fit = output_mega[:n_mega, :]
        output_pos_fit = output_pos[:nrows]

        output_mb_extended = torch.repeat_interleave(output_mb_fit, self.cfg.sequence_length_pos, dim=0)
        output_mb_fit = output_mb_extended[:nrows]

        output_mega_extended = torch.repeat_interleave(output_mega_fit,
                                                       self.cfg.sequence_length_pos * self.cfg.sequence_length_mb,
                                                       dim=0)
        output_mega_fit = output_mega_extended[:nrows]

        concat_reps = torch.cat([output_pos_fit, output_mb_fit, output_mega_fit], 1)
        full_reps = torch.cat([zero_embed, concat_reps], 0)
        return full_reps

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
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def get_embeddings(self, indices):
        """
        get_embeddings(self, indices) -> Array
        Method to get embeddings given indices.
        Runs the indices through the embedding layer.
        Args:
            indices (Array): numpy array of indices for which representations are needed.
        """
        indices = torch.from_numpy(indices).to(torch.int64).to(self.device)
        embeddings = self.pos_embed(indices)
        return embeddings.detach().cpu()

    def train_model(self, optimizer, writer):
        """
        train_model(self, data_loader, criterion, optimizer, writer) -> No return object
        Method to train the model.
        Runs the indices through the forward method. Gets output Hi-C.
        Compares with observed Hi-C. Computes Criterion. Saves trained model.
        Args:
            data_loader (DataLoader): DataLoader containing dataset.
            criterion (Criterion): Loss function
            optimizer (Optimizer): Adam like with learning rate.
            writer (SummaryWriter): Tensorboard SummaryWriter
        """
        device = self.device
        cfg = self.cfg
        num_epochs = cfg.num_epochs

        for epoch in range(num_epochs):
            print("Epoch: {}".format(str(epoch)))
            with torch.autograd.set_detect_anomaly(True):
                self.train()
                epoch_loss = 0.0

                for chr in cfg.chr_train_list:
                    indices, values, nrows = get_data(cfg, chr)
                    cum_pos = get_cumpos(cfg, chr)

                    indices = indices.float().to(device)
                    values = values.float().to(device)

                    "Forward Pass"
                    full_reps = self(indices, nrows)

                    input_pairs = torch.combinations(indices, with_replacement=True)
                    input_pairs = input_pairs.view((-1, self.cfg.mlp_batch_size, 2))

                    batch_loss = 0.0
                    for i in range(input_pairs.shape[0]):
                        input_pair = input_pairs[i].long()

                        loss = self.fullMLP(input_pair, values, cum_pos, full_reps)

                        "Backward and optimize"
                        optimizer.zero_grad()
                        loss.backward()
                        clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                        optimizer.step()

                        batch_loss += loss.item()

                    epoch_loss += batch_loss
                    writer.add_scalar('training loss', loss, epoch * 22 + chr)

                    "save model"
                    torch.save(self.state_dict(), cfg.model_dir + self.model_name + '.pth')

            print('Completed epoch %s' % str(epoch + 1))
            print('Average loss: %s' % (epoch_loss / (values.shape[0] * values.shape[1])))

    def post_processing(self, cfg, ind, val, pred, embed, pred_df, prev_error_list, error_compute, zero_embed):
        """
        post_processing(self, cfg, ind, val, pred, embed, pred_df, prev_error_list, error_compute, zero_embed) -> DataFrame, Array
        Post processing method. Compute error and remove padded indices.
        Args:
            cfg (Config): DataLoader containing dataset.
            ind (Array): Loss function
            val (Array): Adam like with learning rate.
            pred (Array): Tensorboard SummaryWriter
            embed (Array): embeddings
            pred_df (DataFrame): Dataframe to put the columns in
            prev_error_list (Array): Error list from the previous batch for averaging
            error_compute (bool): Boolean for computing error
            zero_embed (bool): Boolean to get zero embed
        """
        seq = cfg.sequence_length
        num_seq = int(np.ceil(len(ind) / seq))
        error_list = None

        "compute error"
        if error_compute:
            error_batch = np.zeros((num_seq, seq))

            for n in range(num_seq):
                ind_temp = ind[n * seq:(n + 1) * seq, :]
                val_temp = val[n * seq:(n + 1) * seq]
                pred_temp = pred[n * seq:(n + 1) * seq]
                idx_temp = np.array(np.where(np.sum(ind_temp, axis=1) == 0))[0]

                if len(idx_temp) == 0:
                    error_batch[n, :] = np.square(val_temp - pred_temp)[:, 0]
                else:
                    val_temp = np.delete(val_temp, idx_temp, axis=0)
                    pred_temp = np.delete(pred_temp, idx_temp, axis=0)
                    error_batch[n, 0:len(val_temp)] = np.square(val_temp - pred_temp)[:, 0]

            error_list = error_batch.mean(axis=0)
            if prev_error_list is None:
                print("first batch")
            else:
                error_list = np.mean((error_list, prev_error_list), axis=0)

        "return zero embed"
        if zero_embed:
            idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
            zero_embed = embed[idx[0]]
        else:
            "remove padded indices"
            idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
            ind = np.delete(ind, idx, axis=0)
            val = np.delete(val, idx, axis=0)
            pred = np.delete(pred, idx, axis=0)
            embed = np.delete(embed, idx, axis=0)

            "make dataframe of indices, values, preds, and embeddings"
            pred_df["i"] = ind[:, 0]
            pred_df["j"] = ind[:, 1]
            pred_df["v"] = val
            pred_df["pred"] = pred
            for n in range(2 * cfg.pos_embed_size):
                pred_df[n] = embed[:, n]

        return pred_df, error_list, zero_embed

    def test(self, data_loader):
        """
        test(self, data_loader) -> tensor, tensor, tensor, DataFrame, Array
        Method to test the given model. Computes error after forward pass.
        Runs post processing to compute error and get embeddings.
        Return
        Args:
            data_loader (DataLoader): DataLoader containing dataset.
        """
        device = self.device
        cfg = self.cfg
        seq = cfg.sequence_length
        error_list = None

        predictions = torch.empty(0, seq).to(device)
        test_error = torch.empty(0).to(device)
        target_values = torch.empty(0, seq).to(device)
        df_columns = ["i", "j", "v", "pred"] + list(np.arange(2 * cfg.pos_embed_size))
        main_pred_df = pd.DataFrame(columns=df_columns)

        with torch.no_grad():
            self.eval()
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                pred_df = pd.DataFrame(columns=df_columns)
                indices = indices.to(device)
                values = values.to(device)

                target_values = torch.cat((target_values, values), 0)

                "forward pass"
                lstm_output, embeddings = self.forward(indices)
                predictions = torch.cat((predictions, lstm_output), 0)

                "compute error"
                error = nn.MSELoss(reduction='none')(lstm_output, values)
                test_error = torch.cat((test_error, error), 0)

                "detach everything for post"
                ind = indices.cpu().detach().numpy().reshape(-1, 2)
                val = values.cpu().detach().numpy().reshape(-1, 1)
                pred = lstm_output.cpu().detach().numpy().reshape(-1, 1)
                embed = embeddings.cpu().detach().numpy().reshape(-1, 2 * cfg.pos_embed_size)

                "compute error and get post processed data and embeddings"
                pred_df, error_list, _ = self.post_processing(cfg, ind, val, pred, embed, pred_df,
                                                              error_list, error_compute=False, zero_embed=False)

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        predictions = torch.reshape(predictions, (-1, 1)).cpu().detach().numpy()
        test_error = torch.reshape(test_error, (-1, 1)).cpu().detach().numpy()
        target_values = torch.reshape(target_values, (-1, 1)).cpu().detach().numpy()

        return predictions, test_error, target_values, main_pred_df, error_list

    def zero_embed(self, data_loader):
        """
        zero_pred(self, data_loader) -> Array
        Method to return the representation for padding
        Args:
            data_loader (DataLoader): DataLoader containing dataset.
        """

        device = self.device
        cfg = self.cfg
        pred_df = None
        error_list = None

        with torch.no_grad():
            self.eval()
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                indices = indices.to(device)
                val = values.to(device)

                "forward pass"
                pred, embeddings = self.forward(indices)

                "detach everything for post"
                ind = indices.cpu().detach().numpy().reshape(-1, 2)
                embed = embeddings.cpu().detach().numpy().reshape(-1, 2 * cfg.pos_embed_size)

                "compute error and get post processed data and embeddings"
                pred_df, error_list, zero_embed = self.post_processing(cfg, ind, val, pred, embed, pred_df,
                                                                       error_list, error_compute=False, zero_embed=True)

                if zero_embed is not None:
                    return zero_embed

    def simple_post(self, indices, ig, pred_df):
        """
        simple_post(self, indices, ig, pred_df) -> DataFrame
        Simple post processing method to remove padded indices and create dataframe
        Return
        Args:
            indices (tensor): Concatenated indices
            ig (tensor): Integrated gradients values
            pred_df (DataFrame): for adding ig and indices
        """

        "remove padded indices"
        ind = indices.cpu().detach().numpy().reshape(-1, 2)
        ig = ig.cpu().detach().numpy().reshape(-1, 2)
        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        ig = np.delete(ig, idx, axis=0)

        "add to dataframe"
        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["ig"] = ig[:, 0]

        return pred_df

    def contactProb(self, values, delta=1e-10):
        """
        contactProb(values, delta) -> Array
        Squishes Hi-C values between 0 and 1.
        Args:
            values (Array): the Hi-C values.
            delta (float): small positive value to avoid divide by 0.
        """
        coeff = np.nan_to_num(1 / (values + delta))
        contact_prob = np.power(1 / np.exp(self.cfg.hic_smoothing), coeff)
        return contact_prob
