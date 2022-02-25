import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
import training.ln_lstm as lstm
from captum.attr import LayerIntegratedGradients


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
        self.hidden_size_lstm = cfg.hidden_size_lstm
        self.gpu_id = 0
        self.model_name = cfg.model_name

        "Initializes ebedding layer"
        self.pos_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size).train()
        nn.init.normal_(self.pos_embed.weight)

        "Initializes LSTM decoder"
        self.lstm = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
        self.out = nn.Linear(cfg.hidden_size_lstm * cfg.sequence_length, cfg.output_size_lstm * cfg.sequence_length)
        self.sigm = nn.Sigmoid()
        self.hidden, self.state = None, None

        "freezes LSTM during training"
        if cfg.lstm_nontrain:
            self.lstm.requires_grad = False
            self.pos_embed.requires_grad = True
            self.out.requires_grad = True

    def forward_with_hidden(self, input, iter):
        """
        forward_with_hidden(self, input, iter) -> tensor, tensor
        Forward method to pass hidden states from previous frame to new frame.
        Args:
            input (Tensor): The concatenated pairwise indices.
            iter (int): Iteration number in training
        """
        if iter == 0:
            self.hidden, self.state = self._initHidden(input.shape[0])
        if input.shape[0] != self.cfg.batch_size:
            self.hidden = self.hidden[:, :input.shape[0], :]
            self.state = self.state[:, :input.shape[0], :]
        embeddings = self.pos_embed(input.long())
        embeddings = embeddings.view((input.shape[0], self.cfg.sequence_length, -1))
        output, (hidden, state) = self.lstm(embeddings, (self.hidden, self.state))
        self.hidden, self.state = hidden.detach(), state.detach()
        output = self.out(output.reshape(input.shape[0], -1))
        output = self.sigm(output)
        return output, embeddings

    def forward(self, input):
        """
        forward_reinit(self, input) -> tensor, tensor
        Default forward method that reinitializes hidden sates in every frame.
        Args:
            input (Tensor): The concatenated pairwise indices.
        """
        hidden, state = self._initHidden(input.shape[0])
        embeddings = self.pos_embed(input.long())
        embeddings = embeddings.view((input.shape[0], self.cfg.sequence_length, -1))
        output, (_, _) = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(input.shape[0], -1))
        output = self.sigm(output)
        return output, embeddings

    def ko_forward(self, embeddings):
        """
        ko_forward(self, embeddings) -> tensor, tensor
        Forward method to be used when doing knockout with manipulated embeddings.
        Args:
            embeddings (Tensor): The embeddings for positions.
        """
        hidden, state = self._initHidden(embeddings.shape[0])
        output, _ = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(embeddings.shape[0], -1))
        output = self.sigm(output)
        return output, embeddings

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
        output, _ = self.decoder_cnn(embeddings)
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
            No Args, specify learning rate in config. Uses Adam as default.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
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

    def train_model(self, data_loader, criterion, optimizer, writer):
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
            print(epoch)
            with torch.autograd.set_detect_anomaly(True):
                self.train()
                running_loss = 0.0
                for i, (indices, values) in enumerate(tqdm(data_loader)):
                    indices = indices.to(device)
                    values = values.to(device)

                    "Forward Pass"
                    output, _ = self.forward(indices)
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
            torch.save(self.state_dict(), cfg.model_dir + self.model_name + '.pth')
            print('Completed epoch %s' % str(epoch + 1))
            print('Average loss: %s' % (running_loss / len(data_loader)))

    def post_processing(self, cfg, ind, val, pred, embed, pred_df, prev_error_list, error_compute):
        """
        post_processing(self, cfg, ind, val, pred, embed, pred_df, prev_error_list, error_compute) -> DataFrame, Array
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

        return pred_df, error_list

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
                pred_df, error_list = self.post_processing(cfg, ind, val, pred, embed, pred_df,
                                                           error_list, error_compute=False)

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        predictions = torch.reshape(predictions, (-1, 1)).cpu().detach().numpy()
        test_error = torch.reshape(test_error, (-1, 1)).cpu().detach().numpy()
        target_values = torch.reshape(target_values, (-1, 1)).cpu().detach().numpy()

        return predictions, test_error, target_values, main_pred_df, error_list

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

    def compute_rowwise_ig(self, main_pred_df):
        """
        compute_rowwise_ig(self, main_pred_df) -> DataFrame
        Method to normalize IG values indice wise and for all indices.
        Return
        Args:
            main_pred_df (DataFrame): DataFrame containing indices and ig values
        """
        
        main_pred_df = main_pred_df.groupby('i').agg({'ig': 'sum'})

        '''
        df_columns = ["i", "ig"]
        final_pred_df = pd.DataFrame(columns=df_columns)
        prev_row = None
        for i in range(len(main_pred_df)):
            row = main_pred_df.iloc[i]["i"]
            if row == prev_row:
                continue

            prev_row = row
            subset_df = main_pred_df.loc[main_pred_df["i"] == row]
            sign = subset_df["ig"] > 0
            new_col = sign * ((subset_df["ig"] - subset_df[sign]["ig"].min()) / subset_df["ig"].max()) + ~sign * -(
                    (subset_df["ig"] - subset_df[~sign]["ig"].max()) / subset_df["ig"].min())
            mean_ig = new_col.sum()
            final_pred_df = final_pred_df.append({'i': row, 'ig': mean_ig}, ignore_index=True)
        '''

        sign = main_pred_df["ig"] > 0
        main_pred_df["ig"] = sign * (
                (main_pred_df["ig"] - main_pred_df[sign]["ig"].min()) / main_pred_df["ig"].max()) \
                              + ~sign * -((main_pred_df["ig"] - main_pred_df[~sign]["ig"].max()) / main_pred_df[
            "ig"].min())
        return main_pred_df

    def get_captum_ig(self, data_loader):
        """
        get_captum_ig(self, data_loader) -> DataFrame
        Method to compute Integrated Gradients score for all indices and normalize them.
        Return
        Args:
            data_loader (DataLoader): DataLoader containing dataset to iterate over
        Raises:
            if there is an error in selection targets concerning tuples. Change some internal code in:
                def _select_targets(output: Tensor, target: TargetType) -> Tensor:
                    num_examples = output[0].shape[0]
                    dims = len(output[0].shape)
                    device = output[0].device

                    elif isinstance(target, list):
                        assert len(target) == num_examples, "Target list length does not match output!"
                        if isinstance(target[0], int):
                            assert dims == 2, "Output must be 2D to select tensor of targets."
                            return torch.gather(
                                output[0], 1, torch.tensor(target, device=device).reshape(len(output[0]), 1)
                            )
        """
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length

        "specify baseline as random"
        input_baseline = torch.rand(1, num_outputs, 2).float() * 1e5
        baseline_stack = (input_baseline.int().to(device))
        df_columns = ["i", "j", "ig"]
        main_pred_df = pd.DataFrame(columns=df_columns)

        for i, (indices, values) in enumerate(tqdm(data_loader)):
            pred_df = pd.DataFrame(columns=df_columns)
            indices = torch.tensor(indices, requires_grad=True)
            input_stack = (indices.to(device))

            "specify targets as all rows"
            ig_target = list(np.arange(len(indices)))
            ig_target = [int(x) for x in ig_target]

            "compute layer integrated gradients for embedding layer"
            ig = LayerIntegratedGradients(self, self.pos_embed)
            attributions, delta = ig.attribute(input_stack, baseline_stack, target=ig_target,
                                               return_convergence_delta=True, attribute_to_layer_input=False)

            "sum attributions for all dimensions"
            attributions = torch.sum(attributions, 3)
            pred_df = self.simple_post(indices, attributions, pred_df)
            main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        "normalize computed IG values"
        main_pred_df = main_pred_df.reset_index(drop=True)
        main_pred_df = self.compute_rowwise_ig(main_pred_df)
        return main_pred_df

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

    def inverse_exp(self, preds, delta=1e-10):
        """
        inverse_exp(preds, delta) -> Array
        Given squished Hi-C values between 0 and 1, computes observed value.
        Args:
            preds (Array): the Hi-C predictions.
            delta (float): small positive value to avoid divide by 0.
        """
        coeffs = -1 / 8 * np.log(preds)
        inv_exp_val = np.nan_to_num(1 / coeffs) - delta
        return inv_exp_val

    def ko_post(self, ind, val, pred, pred_df, mode):
        """
        ko_post(ind, val, pred, pred_df, mode) -> DataFrame
        Removes padded indices and accounts for confusion during duplication.
        Args:
            ind (Array): Concatenated indices
            val (Array): The Hi-C observed values
            pred (Array): The Hi-C predicted values
            pred_df (DataFrame): Dataframe to add knockout values with indices
            mode (string): Can specify whether running duplication or not. If not running
                            duplication, performs simple post processing
        """

        "remove padded indices"
        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        val = np.delete(val, idx, axis=0)
        pred = np.delete(pred, idx, axis=0)
        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["v"] = val
        pred_df["ko_pred"] = pred

        "account for confusion in the reads when doing duplication"
        if mode == "dup":
            start = int(pred_df['i'].min())
            stop = int(pred_df['i'].max())
            chunk_start = self.cfg.chunk_start
            chunk_end = self.cfg.chunk_end
            dupl_start = self.cfg.dupl_start
            dupl_end = self.cfg.dupl_end
            shift = self.cfg.shift

            for r in range(start, stop + 1):
                if r < chunk_start:
                    continue
                elif r >= chunk_start and r <= chunk_end:
                    og_pred = pred_df.loc[pred_df['i'] == r]["ko_pred"]
                    dupl_pred = pred_df.loc[pred_df['i'] == r + shift]["ko_pred"]

                    if len(dupl_pred) == 0:
                        continue

                    "compute inverse exponentials"
                    og_inv = self.inverse_exp(og_pred)
                    dupl_inv = self.inverse_exp(dupl_pred)

                    "squish after adding again"
                    exp_pred = self.contactProb(og_inv + dupl_inv)
                    pred_df.loc[pred_df['i'] == r, 'ko_pred'] = exp_pred

                elif r >= dupl_start and r <= dupl_end:
                    pred_df.loc[pred_df['i'] == r, 'ko_pred'] = pred_df.loc[pred_df['i'] == r + shift]["ko_pred"]

                elif r > dupl_end:
                    break

        return pred_df

    def perform_ko(self, data_loader, embed_rows, start, mode):
        """
        perform_ko(data_loader, embed_rows, start, mode) -> tensor, DataFrame
        Performs knockout.
        Args:
            data_loader (DataLoader): DataLoader containing dataset
            embed_rows (Array): embeddings
            start (int):  shift value
            mode (string): Can specify whether running duplication or not. If not running
                            duplication, performs simple post processing
        """
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length
        ko_predictions = torch.empty(0, num_outputs).to(device)
        main_pred_df = pd.DataFrame(columns=["i", "j", "v", "ko_pred"])

        with torch.no_grad():
            self.eval()
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                pred_df = pd.DataFrame(columns=["i", "j", "v", "ko_pred"])
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
                embeddings = embeddings.view((-1, self.cfg.sequence_length, 2 * cfg.pos_embed_size)).float().to(device)

                "run forward with updated embeddings"
                lstm_output, _ = self.ko_forward(embeddings)
                ko_predictions = torch.cat((ko_predictions, lstm_output), 0)

                "run postprocessing"
                pred = lstm_output.cpu().detach().numpy().reshape(-1, 1)
                pred_df = self.ko_post(ind, val, pred, pred_df, mode=mode)

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        ko_predictions = torch.reshape(ko_predictions, (-1, 1)).cpu().detach().numpy()

        return ko_predictions, main_pred_df
