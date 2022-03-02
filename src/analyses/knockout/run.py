import torch
import numpy as np
import os
import pandas as pd
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
import matplotlib.pyplot as plt
from training.data_utils import get_samples_sparse, get_cumpos
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from training.config import Config
from training.test_model import test_model
from analyses.feature_attribution.tf import TFChip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Knockout():
    """
    Class to perform knockout experiments.
    Includes methods that help you do knockout.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.chr = chr
        self.cell = cfg.cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.hek_file = cfg.hic_path + cfg.cell + "/HEK239T-WT.matrix"
        self.tal1ko_file = cfg.hic_path + cfg.cell + "/Tal1KO.matrix"
        self.lmo2ko_file = cfg.hic_path + cfg.cell + "/Lmo2KO.matrix"

    def alter_index(self, embed_rows):
        """

        """
        if self.chr == 1:
            cum_pos = 0
        else:
            key = "chr" + str(self.chr - 1)
            cum_pos = int(self.sizes[key])

        embed_rows["new_index"] = embed_rows.index + cum_pos
        embed_rows["new_index"] = embed_rows["new_index"].astype(int)
        embed_rows.index = embed_rows['new_index']
        embed_rows = embed_rows.drop(columns=['new_index'])

        return embed_rows

    def get_ctcf_indices(self):
        """

        """

        ctcf_ob = TFChip(cfg, chr)
        data = ctcf_ob.get_ctcf_data()
        data = data.filter(['start'], axis=1)
        cum_pos = get_cumpos(self.cfg, self.chr)
        data["start"] = data["start"] + cum_pos
        return data

    def get_trained_representations(self, method="hiclstm"):
        """
        get_trained_representations(method) -> Array, int
        Gets fully trained representations for given method, cell type and chromosome.
        obtain sniper and sca representations from respective methods.
        Should contain SNIPER and SCA positions end internal representations.
        Args:
            method (string): one of hiclstm, sniper, sca
        """

        pred_data = pd.read_csv(
            self.cfg.output_directory + "%s_%s_predictions_chr%s.csv" % (method, self.cell, str(self.chr)),
            sep="\t")
        pred_data = pred_data.drop(['Unnamed: 0'], axis=1)
        representations, start, stop = self.convert_df_to_np(pred_data, method=method)
        return representations, start, stop, pred_data

    def convert_df_to_np(self, pred_data, method="hiclstm"):
        """

        """

        i_start = int(pred_data['i'].min())
        i_stop = int(pred_data['i'].max())
        j_start = int(pred_data['j'].min())
        j_stop = int(pred_data['j'].max())

        if i_start < j_start:
            start = i_start
        else:
            start = j_start

        if i_stop > j_stop:
            stop = i_stop
        else:
            stop = j_stop

        try:
            embed_rows = np.load(
                self.cfg.output_directory + "%s_rep_%s_chr%s.npy" % (method, self.cfg.cell, str(self.chr)))
        except:
            nrows = int(stop - start)
            embed_rows = np.zeros((nrows + 1, self.cfg.pos_embed_size))

            i_old = 0
            j_old = 0
            for r in range(len(pred_data)):
                i_new = int(pred_data.loc[r, "i"])
                if i_new == i_old:
                    continue
                else:
                    i_old = i_new
                    if np.all((embed_rows[i_new - start, :] == 0)):
                        col = list(np.arange(self.cfg.pos_embed_size))
                        col = [str(x) for x in col]
                        embed_rows[i_new - start, :] = np.array(pred_data.loc[r, col])

                j_new = int(pred_data.loc[r, "j"])

                if j_new == j_old:
                    continue
                else:
                    j_old = j_new
                    if np.all((embed_rows[j_new - start, :] == 0)):
                        col = list(np.arange(self.cfg.pos_embed_size, 2 * self.cfg.pos_embed_size))
                        col = [str(x) for x in col]
                        embed_rows[j_new - start, :] = np.array(pred_data.loc[r, col])

        return embed_rows, start, stop

    def ko_representations(self, representations, start, indices, mode="average"):
        """

        """
        window = self.cfg.ko_window

        for ind in indices:
            if mode == "average":
                if ind - start - window < 0 or ind - start + window > len(representations):
                    window = int(window // 2)

                window_left_arr = representations[ind - start - window: ind - start, :].copy()
                window_right_arr = representations[ind - start + 1: ind - start + window + 1, :].copy()

                window_arr_avg = np.stack((window_left_arr, window_right_arr)).mean(axis=0).mean(axis=0)
                representations[ind - start, :] = window_arr_avg
            elif mode == "zero":
                representations[ind - start, :] = np.zeros((1, self.cfg.pos_embed_size))
        return representations

    def compute_kodiff(self, pred_data, ko_pred_df, indices, stop):
        """

        """

        indices = np.array(indices[:5])
        diff_list = np.zeros((len(indices), 11))
        for i, ind in enumerate(indices):
            for k in np.arange(0, 101, 10):
                subset_og = pred_data.loc[pred_data["i"] == ind + k]
                if subset_og.empty or (ind + k) > stop:
                    continue
                subset_ko = ko_pred_df.loc[ko_pred_df["i"] == ind + k]
                mean_diff = np.mean(np.array(subset_ko["ko_pred"]) - np.array(subset_og["pred"]))
                diff_list[i, int(k/10)] = mean_diff

        mean_diff = np.mean(diff_list, axis=0)
        return mean_diff

    def perform_ko(self, model):
        """

        """

        "get knockout indices depending on experiment"
        if self.cfg.ko_experiment == "ctcf":
            if self.cfg.ctcf_indices == "all":
                indices = self.get_ctcf_indices()
            else:
                indices = self.cfg.ctcf_indices_22
        elif self.cfg.ko_experiment == "foxg1":
            indices = [222863]

        "load data"
        data_loader = get_data_loader_chr(self.cfg, self.chr)

        "get representations"
        representations, start, stop, pred_data = self.get_trained_representations(method="hiclstm")

        "alter representations"
        representations = self.ko_representations(representations, start, indices, mode=cfg.ko_mode)

        "run through model using altered representations, save ko predictions"
        _, ko_pred_df = model.perform_ko(data_loader, representations, start, mode="ko")
        ko_pred_df.to_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        "compute difference between WT and KO predictions"
        mean_diff = self.compute_kodiff(pred_data, ko_pred_df, indices, stop)
        return mean_diff

    def normalize_embed(self, representations):
        """

        """

        "normalize representations"
        for n in range(len(representations)):
            norm = np.linalg.norm(representations[n, :])
            if norm == 0:
                continue
            else:
                representations[n, :] = representations[n, :] / norm
        return representations

    def normalize_embed_predict(self, model):
        """

        """

        "load data"
        data_loader = get_data_loader_chr(self.cfg, self.chr)

        "get representations"
        representations, start, stop, pred_data = self.get_trained_representations(method="hiclstm")

        "alter representations"
        representations = self.normalize_embed(representations)

        "run through model using altered representations, save ko predictions"
        _, ko_pred_df = model.perform_ko(data_loader, representations, start, mode="ko")
        ko_pred_df.to_csv(cfg.output_directory + "hiclstm_%s_norm_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

    def change_index(self, list_split):
        """

        """
        temp = [k.split('|')[-1] for k in list_split]
        chr_list = []
        index_list = []
        for t in temp:
            index = t.split(':')
            chr_list.append(index[0])
            index_list.append(index[1].split('-'))

        loc_list = []
        for ind in index_list:
            loc = int(((int(ind[0]) + int(ind[1])) / 2) // 10000)
            loc_list.append(loc)

        return loc_list, chr_list

    def tal_lmo2_preprocess(self):
        """

        """
        # hek_mat = pd.read_csv(self.hek_file, sep="\t")
        hek_mat = pd.read_csv(self.lmo2ko_file, sep="\t")

        index, chr_list = self.change_index(list(hek_mat.index))
        columns, _ = self.change_index(hek_mat.columns)

        hek_mat.index = index
        hek_mat.columns = columns
        hek_mat["chr"] = chr_list

        tal1_mat = hek_mat.loc[hek_mat["chr"] == "chr1"]
        tal1_mat = tal1_mat.iloc[:, 0:285]
        lmo2_mat = hek_mat.loc[hek_mat["chr"] == "chr11"]
        lmo2_mat = lmo2_mat.iloc[:, 286:632]
        tal1_mat = tal1_mat.groupby(level=0, axis=1).sum()
        tal1_mat = tal1_mat.groupby(level=0, axis=0).sum()
        lmo2_mat = lmo2_mat.groupby(level=0, axis=1).sum()
        lmo2_mat = lmo2_mat.groupby(level=0, axis=0).sum()

        tal_i = list(tal1_mat.index)
        tal_j = tal1_mat.columns
        lmo2_i = list(lmo2_mat.index)
        lmo2_j = lmo2_mat.columns

        tal_df = pd.DataFrame(columns=["i", "j", "v"])
        for i in tal_i:
            for j in tal_j:
                tal_df = tal_df.append({"i": i, "j": j, "v": tal1_mat.loc[i][j]}, ignore_index=True)

        lmo2_df = pd.DataFrame(columns=["i", "j", "v"])
        for i in lmo2_i:
            for j in lmo2_j:
                lmo2_df = lmo2_df.append({"i": i, "j": j, "v": lmo2_mat.loc[i][j]}, ignore_index=True)

        tal_df.to_csv(cfg.hic_path + cfg.cell + "/tal_df.txt", sep="\t")
        lmo2_df.to_csv(cfg.hic_path + cfg.cell + "/lmo2_df.txt", sep="\t")

        return tal_df, lmo2_df

    def prepare_tal1_lmo2(self, cfg):
        """

        """
        tal_df = pd.read_csv(cfg.hic_path + cfg.cell + "/tal_df.txt", sep="\t")
        lmo2_df = pd.read_csv(cfg.hic_path + cfg.cell + "/lmo2_df.txt", sep="\t")

        tal_df = tal_df.drop(['Unnamed: 0'], axis=1)
        lmo2_df = lmo2_df.drop(['Unnamed: 0'], axis=1)

        tal_df[['i', 'j']] = tal_df[['i', 'j']].astype('int64')
        lmo2_df[['i', 'j']] = lmo2_df[['i', 'j']].astype('int64')

        values = torch.empty(0, cfg.sequence_length)
        input_idx = torch.empty(0, cfg.sequence_length, 2)
        sample_index_list = []

        input_idx_tal1, values_tal1, sample_index_tal1 = get_samples_sparse(tal_df, 1, cfg)
        values_tal1 = F.pad(input=values_tal1, pad=(0, 4, 0, 0), mode='constant', value=0)
        input_idx_tal1 = F.pad(input=input_idx_tal1, pad=(0, 0, 0, 4, 0, 0), mode='constant', value=0)

        values = torch.cat((values, values_tal1.float()), 0)
        input_idx = torch.cat((input_idx, input_idx_tal1), 0)
        sample_index_list.append(sample_index_tal1)

        input_idx_lmo2, values_lmo2, sample_index_lmo2 = get_samples_sparse(lmo2_df, 11, cfg)
        values = torch.cat((values, values_lmo2.float()), 0)
        input_idx = torch.cat((input_idx, input_idx_lmo2), 0)
        sample_index_list.append(sample_index_lmo2)

        sample_index_tensor = np.vstack(sample_index_list)
        # create dataset, dataloader
        dataset = torch.utils.data.TensorDataset(input_idx, values)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

        return data_loader, sample_index_tensor

    def train_tal1_lmo2(self, model, cfg, model_name):
        """

        """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter('./tensorboard_logs/' + model_name + timestr)

        optimizer, criterion = model.compile_optimizer(cfg)
        data_loader, samples = self.prepare_tal1_lmo2(cfg)

        model.train_model(data_loader, criterion, optimizer, writer)
        torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')
        pass

    def test_tal1_lmo2(self, model, cfg):
        """

        """
        data_loader, samples = self.prepare_tal1_lmo2(cfg)
        predictions, test_error, values, pred_df, error_list = model.test(data_loader)

        pred_df.to_csv(cfg.output_directory + "%s_predictions_chr.csv" % (cfg.cell), sep="\t")
        pass


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell

    "load model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    if cfg.ko_experiment == "foxg1":
        cfg.chr_test_list = cfg.foxg1_chr

    for chr in cfg.chr_test_list:
        print('Knockout Start Chromosome: {}'.format(chr))
        ko_ob = Knockout(cfg, chr)

        if cfg.ko_compute_test:
            "run test if predictions not computed yet"
            test_model(model, cfg, chr)

        if cfg.perform_ko:
            "perform ko"
            mean_diff = ko_ob.perform_ko(model)
        elif cfg.load_ko:
            ko_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cell, str(chr)), sep="\t")
        elif cfg.normalize_embed:
            ko_ob.normalize_embed_predict(model)

        # tal_data, lmo2_data = ko_ob.tal_lmo2_preprocess()
        # ko_ob.train_tal1_lmo2(model, cfg, model_name)
        # ko_ob.test_tal1_lmo2(model, cfg)
