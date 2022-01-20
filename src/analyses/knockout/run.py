from __future__ import division
import torch
import numpy as np
import os
import pandas as pd
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
import matplotlib.pyplot as plt
from training.data_utils import get_samples_sparse
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'
mode = "test"


class Knockout():

    def __init__(self, cfg, cell, chr):
        self.cfg = cfg
        self.chr = chr
        self.cell = cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.file_path = os.path.join(cfg.downstream_dir, "ctcf")
        self.hek_file = cfg.hic_path + cfg.cell + "/HEK239T-WT.matrix"

    def alter_index(self, embed_rows):
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
        data = pd.read_csv(os.path.join(self.file_path, "chr" + str(chr) + ".bed"), sep="\t", header=None)
        column_list = ["chr", "start", "end", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]
        data.columns = column_list
        data = data.filter(['start'], axis=1)
        data["start"] = (data["start"]).astype(int) // self.res

        if self.chr == 1:
            cum_pos = 0
        else:
            key = "chr" + str(self.chr - 1)
            cum_pos = int(self.sizes[key])

        data["start"] = data["start"] + cum_pos
        data = data.sort_values('start')

        return data

    def convert_df_to_np(self, pred_data):
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

        nrows = int(stop - start)

        embed_rows = np.zeros((nrows + 1, cfg.pos_embed_size))

        i_old = 0
        j_old = 0
        for r in range(len(pred_data)):
            i_new = int(pred_data.loc[r, "i"])
            if i_new == i_old:
                continue
            else:
                i_old = i_new
                if np.all((embed_rows[i_new - start, :] == 0)):
                    col = list(np.arange(cfg.pos_embed_size))
                    col = [str(x) for x in col]
                    embed_rows[i_new - start, :] = np.array(pred_data.loc[r, col])

            j_new = int(pred_data.loc[r, "j"])

            if j_new == j_old:
                continue
            else:
                j_old = j_new
                if np.all((embed_rows[j_new - start, :] == 0)):
                    col = list(np.arange(cfg.pos_embed_size, 2 * cfg.pos_embed_size))
                    col = [str(x) for x in col]
                    embed_rows[j_new - start, :] = np.array(pred_data.loc[r, col])

        return embed_rows, start, stop

    def ko_indices(self, embed_rows, start, indices):
        # chose from input ko indices or give your own
        window = 10

        for ind in indices:
            if ind - start - window < 0 or ind - start + window > len(embed_rows):
                window = int(window // 2)

            window_left_arr = embed_rows[ind - start - window: ind - start, :].copy()
            window_right_arr = embed_rows[ind - start + 1: ind - start + window + 1, :].copy()

            window_arr_avg = np.stack((window_left_arr, window_right_arr)).mean(axis=0).mean(axis=0)
            embed_rows[ind - start, :] = window_arr_avg
        return embed_rows

    def compute_kodiff(self, pred_data, ko_pred_df, indices, stop):
        indices = np.array(indices[:5])
        diff_list = np.zeros((len(indices), 11))
        for i, ind in enumerate(indices):
            for k in range(11):
                subset_og = pred_data.loc[pred_data["i"] == ind[0] + k]
                if subset_og.empty or (ind[0] + k) > stop:
                    continue
                subset_ko = ko_pred_df.loc[ko_pred_df["i"] == ind[0] + k]
                mean_diff = np.mean(np.array(subset_ko["ko_pred"]) - np.array(subset_og["pred"]))
                diff_list[i, k] = mean_diff

        mean_diff = np.mean(diff_list, axis=0)

        pos = np.linspace(0, 1, 11)
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Distance between positions in Mbp", fontsize=20)
        plt.ylabel("Average Difference in Contact Strength \n (KO - No KO)", fontsize=20)
        plt.plot(pos, mean_diff, marker='o', markersize=16, color="C0", linewidth=3, label="CTCF KO")
        plt.legend(fontsize=18)
        plt.show()

        return mean_diff

    def perform_ko(self, model, pred_data, indices):
        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        embed_rows, start, stop = self.convert_df_to_np(pred_data)
        embed_rows = self.ko_indices(embed_rows, start, indices)

        _, ko_pred_df = model.perform_ko(data_loader, embed_rows, start, mode="ko")
        ko_pred_df.to_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        mean_diff = self.compute_kodiff(pred_data, ko_pred_df, indices, stop)

        return ko_pred_df, mean_diff

    def normalize_embed(self, embed_rows):
        for n in range(len(embed_rows)):
            norm = np.linalg.norm(embed_rows[n, :])
            if norm == 0:
                continue
            else:
                embed_rows[n, :] = embed_rows[n, :] / norm
        return embed_rows

    def normalize_embed_predict(self, model, pred_data):
        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        embed_rows, start, stop = self.convert_df_to_np(pred_data)
        embed_rows = self.normalize_embed(embed_rows)

        _, ko_pred_df = model.perform_ko(data_loader, embed_rows, start, mode="ko")
        ko_pred_df.to_csv(cfg.output_directory + "shuffle_%s_norm_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        return ko_pred_df

    def change_index(self, list_split):
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
        hek_mat = pd.read_csv(self.hek_file, sep="\t")
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
        timestr = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter('./tensorboard_logs/' + model_name + timestr)

        optimizer, criterion = model.compile_optimizer(cfg)
        data_loader, samples = self.prepare_tal1_lmo2(cfg)

        model.train_model(data_loader, criterion, optimizer, writer)
        torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')
        pass

    def test_tal1_lmo2(self, model, cfg):
        data_loader, samples = self.prepare_tal1_lmo2(cfg)
        predictions, test_error, values, pred_df, error_list = model.test(data_loader)

        pred_df.to_csv(cfg.output_directory + "%s_predictions_chr.csv" % (cfg.cell), sep="\t")
        pass

    def tf_ko(self, model, pred_data):
        ctcf_indices_21 = [279219, 279229]
        ctcf_indices_22 = [284706, 284743]

        znf143_chr = 11
        znf143_indices = [182532]

        foxg1_chr = 14
        foxg1_indices = [222863]

        self.perform_ko(model, pred_data, foxg1_indices)

        pass


if __name__ == '__main__':

    cfg = config.Config()
    cell = cfg.cell

    # load model
    model_name = "shuffle_" + cell
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    # test_chr = list(range(21, 23))
    test_chr = [14]

    for chr in test_chr:
        print('Testing Start Chromosome: {}'.format(chr))
        pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        # ko_pred_df = pd.read_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (cell, str(chr)), sep="\t")
        ko_ob = Knockout(cfg, cell, chr)

        # indices = ko_ob.get_ctcf_indices()
        # ko_pred_df, mean_diff = ko_ob.perform_ko(model, pred_data, indices)
        # ko_pred_df = ko_ob.normalize_embed_predict(model, pred_data)

        # tal_data, lmo2_data = ko_ob.tal_lmo2_preprocess()
        # ko_ob.train_tal1_lmo2(model, cfg, model_name)
        # ko_ob.test_tal1_lmo2(model, cfg)

        ko_ob.tf_ko(model, pred_data)



    print("done")
