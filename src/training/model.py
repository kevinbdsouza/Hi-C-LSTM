import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
import training.lstm as lstm
from captum.attr import LayerIntegratedGradients


class SeqLSTM(nn.Module):
    def __init__(self, cfg, device, model_name):
        super(SeqLSTM, self).__init__()
        self.device = device
        self.cfg = cfg
        self.hidden_size_lstm = cfg.hidden_size_lstm
        self.gpu_id = 0
        self.model_name = model_name

        self.pos_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size).train()
        nn.init.normal_(self.pos_embed.weight)

        self.lstm = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
        self.out = nn.Linear(cfg.hidden_size_lstm * cfg.sequence_length, cfg.output_size_lstm * cfg.sequence_length)
        self.sigm = nn.Sigmoid()

        if cfg.lstm_nontrain:
            self.lstm.requires_grad = False
            self.pos_embed.requires_grad = True
            self.out.requires_grad = True

    def forward2(self, input, iter):
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

    def forward1(self, input):
        hidden, state = self._initHidden(input.shape[0])
        embeddings = self.pos_embed(input.long())
        embeddings = embeddings.view((input.shape[0], self.cfg.sequence_length, -1))
        output, (hidden, state) = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(input.shape[0], -1))
        output = self.sigm(output)
        return output, embeddings

    def forward(self, input):
        hidden, state = self._initHidden(input.shape[0])
        embeddings = self.pos_embed(input.long())
        embeddings = embeddings.view((input.shape[0], self.cfg.sequence_length, -1))
        output, (hidden, state) = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(input.shape[0], -1))
        output = self.sigm(output)
        return output

    def ko_forward(self, embeddings):
        hidden, state = self._initHidden(embeddings.shape[0])
        output, _ = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(embeddings.shape[0], -1))
        output = self.sigm(output)
        return output, embeddings

    def _initHidden(self, batch_size):
        h = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        c = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)

        return h, c

    def compile_optimizer(self, cfg):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        return optimizer, criterion

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.model_name + '.pth'))
            # self.load_state_dict(torch.load("/home/kevindsouza/Downloads/" + self.model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def get_embeddings(self, indices):
        device = self.device
        indices = torch.from_numpy(indices).to(torch.int64).to(device)
        embeddings = self.pos_embed(indices)
        return embeddings.detach().cpu()

    def train_model(self, data_loader, criterion, optimizer, writer):
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

                    # Forward pass
                    output, _ = self.forward1(indices)
                    loss = criterion(output, values)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                    optimizer.step()

                    running_loss += loss.item()
                    writer.add_scalar('training loss',
                                      loss, i + epoch * len(data_loader))

            torch.save(self.state_dict(), cfg.model_dir + self.model_name + '.pth')
            print('Completed epoch %s' % str(epoch + 1))
            print('Average loss: %s' % (running_loss / len(data_loader)))

    def post_processing(self, i, cfg, ind, val, pred, embed, pred_df, prev_error_list, window_model, error_compute):
        seq = cfg.sequence_length
        num_seq = int(np.ceil(len(ind) / seq))
        error_list = None

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

        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        val = np.delete(val, idx, axis=0)
        pred = np.delete(pred, idx, axis=0)
        if i == 0:
            zero_embed = embed[idx[0], :]
        embed = np.delete(embed, idx, axis=0)

        if window_model:
            start = int(ind[0][0])
            stop = int(ind[len(ind) - 1][0])
            ind_df = pd.DataFrame(ind, columns=["i", "j"])

            del_index = []
            replace_index = []
            alternate_index = []
            for r in range(start, stop + 1):
                temp = ind_df.loc[ind_df['i'] == r]

                j_prev = r
                for index, row in temp.iterrows():
                    if row["j"] < j_prev:
                        num_round = int(np.floor(index / 150))
                        break
                    else:
                        j_prev = row["j"]

                for k in range(num_round):
                    del_index.append(list(np.arange(index + k * 150, index + k * 150 + 75)))
                    if k == num_round - 1:
                        alternate_index.append(list(np.arange(index + k * 150 + 75, len(temp) - 1)))
                        replace_index.append(list(np.arange((k + 1) * 150, index)))
                    alternate_index.append(list(np.arange(index + k * 150 + 75, index + k * 150 + 150)))
                    replace_index.append(list(np.arange((k + 1) * 150, (k + 1) * 150 + 75)))
                    if len(alternate_index[-1]) != len(replace_index[-1]):
                        print("pause")

            replace_index = [item for sublist in replace_index for item in sublist]
            alternate_index = np.array([item for sublist in alternate_index for item in sublist])
            del_index = np.array([item for sublist in del_index for item in sublist])
            final_delete_array = np.concatenate([del_index, alternate_index])

            for n in range(len(replace_index)):
                ind[replace_index[n], :] = ind[alternate_index[n], :]
                val[replace_index[n]] = val[alternate_index[n], :]
                pred[replace_index[n]] = pred[alternate_index[n]]
                embed[replace_index[n], :] = embed[alternate_index[n], :]

            ind = np.delete(ind, final_delete_array, axis=0)
            val = np.delete(val, final_delete_array, axis=0)
            pred = np.delete(pred, final_delete_array, axis=0)
            embed = np.delete(embed, final_delete_array, axis=0)

        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["v"] = val
        pred_df["pred"] = pred
        for n in range(2 * cfg.pos_embed_size):
            pred_df[n] = embed[:, n]

        return pred_df, error_list

    def test(self, data_loader):
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
            # hidden, state = self._initHidden(cfg.batch_size)
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                pred_df = pd.DataFrame(columns=df_columns)
                indices = indices.to(device)
                values = values.to(device)

                target_values = torch.cat((target_values, values), 0)

                # lstm_output, hidden, state, embeddings = self.forward(indices, hidden, state)
                lstm_output, embeddings = self.forward1(indices)
                predictions = torch.cat((predictions, lstm_output), 0)

                error = nn.MSELoss(reduction='none')(lstm_output, values)
                test_error = torch.cat((test_error, error), 0)

                ind = indices.cpu().detach().numpy().reshape(-1, 2)
                val = values.cpu().detach().numpy().reshape(-1, 1)
                pred = lstm_output.cpu().detach().numpy().reshape(-1, 1)
                embed = embeddings.cpu().detach().numpy().reshape(-1, 2 * cfg.pos_embed_size)

                pred_df, error_list = self.post_processing(i, cfg, ind, val, pred, embed, pred_df,
                                                           error_list, window_model=False, error_compute=False)

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        predictions = torch.reshape(predictions, (-1, 1)).cpu().detach().numpy()
        test_error = torch.reshape(test_error, (-1, 1)).cpu().detach().numpy()
        target_values = torch.reshape(target_values, (-1, 1)).cpu().detach().numpy()

        return predictions, test_error, target_values, main_pred_df, error_list

    def simple_post(self, indices, values, ig, pred_df):

        ind = indices.cpu().detach().numpy().reshape(-1, 2)
        val = values.cpu().detach().numpy().reshape(-1, 1)
        ig = ig.cpu().detach().numpy().reshape(-1, 1)

        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        ig = np.delete(ig, idx, axis=0)

        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["ig"] = ig

        return pred_df

    def compute_rowwise_ig(self, main_pred_df):
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

        sign = final_pred_df["ig"] > 0
        final_pred_df["ig"] = sign * (
                (final_pred_df["ig"] - final_pred_df[sign]["ig"].min()) / final_pred_df["ig"].max()) \
                              + ~sign * -((final_pred_df["ig"] - final_pred_df[~sign]["ig"].max()) / final_pred_df[
            "ig"].min())
        return final_pred_df

    def get_captum_ig(self, data_loader):
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length

        input_baseline = torch.rand(1, num_outputs, 2).float() * 1e5
        baseline_stack = (input_baseline.int().to(device))
        df_columns = ["i", "j", "ig"]
        main_pred_df = pd.DataFrame(columns=df_columns)

        for i, (indices, values) in enumerate(tqdm(data_loader)):
            pred_df = pd.DataFrame(columns=df_columns)
            indices = torch.tensor(indices, requires_grad=True)
            input_stack = (indices.to(device))
            ig_target = list(np.arange(len(indices)))
            ig_target = [int(x) for x in ig_target]
            ig = LayerIntegratedGradients(self, self.pos_embed)
            attributions, delta = ig.attribute(input_stack, baseline_stack, target=ig_target,
                                               return_convergence_delta=True, attribute_to_layer_input=False)

            attributions = torch.sum(attributions[:, :, 0, :], 2)
            pred_df = self.simple_post(indices, values, attributions, pred_df)
            main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        main_pred_df = main_pred_df.reset_index(drop=True)
        main_pred_df = self.compute_rowwise_ig(main_pred_df)
        return main_pred_df

    def contactProb(self, values, delta=1e-10):
        coeff = np.nan_to_num(1 / (values + delta))
        CP = np.power(1 / np.exp(8), coeff)
        return CP

    def inverse_exp(self, preds, delta=1e-10):
        coeffs = -1 / 8 * np.log(preds)
        inv_exp_val = np.nan_to_num(1 / coeffs) - delta
        return inv_exp_val

    def ko_post(self, ind, val, pred, pred_df, mode="ko"):
        idx = np.array(np.where(np.sum(ind, axis=1) == 0))[0]
        ind = np.delete(ind, idx, axis=0)
        val = np.delete(val, idx, axis=0)
        pred = np.delete(pred, idx, axis=0)
        pred_df["i"] = ind[:, 0]
        pred_df["j"] = ind[:, 1]
        pred_df["v"] = val
        pred_df["ko_pred"] = pred

        if mode == "duplication":
            start = int(pred_df['i'].min())
            stop = int(pred_df['i'].max())
            chunk_start = 256803
            chunk_end = 257017
            dupl_start = 257018
            dupl_end = 257232
            shift = 215

            for r in range(start, stop + 1):
                if r < chunk_start:
                    continue
                elif r >= chunk_start and r <= chunk_end:
                    og_pred = pred_df.loc[pred_df['i'] == r]["ko_pred"]
                    dupl_pred = pred_df.loc[pred_df['i'] == r + shift]["ko_pred"]

                    if len(dupl_pred) == 0:
                        continue
                    og_inv = self.inverse_exp(og_pred)
                    dupl_inv = self.inverse_exp(dupl_pred)

                    exp_pred = self.contactProb(og_inv + dupl_inv)
                    pred_df.loc[pred_df['i'] == r, 'ko_pred'] = exp_pred

                elif r >= dupl_start and r <= dupl_end:
                    pred_df.loc[pred_df['i'] == r, 'ko_pred'] = pred_df.loc[pred_df['i'] == r + shift]["ko_pred"]

                elif r > dupl_end:
                    break

        return pred_df

    def perform_ko(self, data_loader, embed_rows, start):
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
                    if ind[n, 0] == 0 and ind[n, 1] == 0:
                        embed_ij[n, 0:cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                        embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = np.mean(embed_rows, axis=0)
                    else:
                        embed_ij[n, 0:cfg.pos_embed_size] = embed_rows[int(ind[n, 0]) - start]
                        embed_ij[n, cfg.pos_embed_size:2 * cfg.pos_embed_size] = embed_rows[int(ind[n, 1]) - start]

                embeddings = torch.from_numpy(embed_ij)
                embeddings = embeddings.view((-1, self.cfg.sequence_length, 2 * cfg.pos_embed_size)).float().to(device)
                lstm_output, _ = self.ko_forward(embeddings)
                ko_predictions = torch.cat((ko_predictions, lstm_output), 0)

                pred = lstm_output.cpu().detach().numpy().reshape(-1, 1)
                pred_df = self.ko_post(ind, val, pred, pred_df, mode="ko")

                main_pred_df = pd.concat([main_pred_df, pred_df], axis=0)

        ko_predictions = torch.reshape(ko_predictions, (-1, 1)).cpu().detach().numpy()

        return ko_predictions, main_pred_df
