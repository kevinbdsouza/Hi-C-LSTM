from __future__ import division
import torch
import numpy as np
import os
import pandas as pd
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr

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

        return embed_rows, start

    def ko_indices(self, embed_rows, start, indices):
        # chose from input ko indices or give your own
        #indices = [279219, 279229]
        indices = [284706, 284743]
        window = 10

        for ind in indices:
            if ind - start - window < 0 or ind - start + window > len(embed_rows):
                window = int(window // 2)

            window_left_arr = embed_rows[ind - start - window: ind - start, :].copy()
            window_right_arr = embed_rows[ind - start + 1: ind - start + window + 1, :].copy()

            # idx_l = np.array(np.where(np.sum(window_left_arr, axis=1) == 0))[0]
            # idx_r = np.array(np.where(np.sum(window_right_arr, axis=1) == 0))[0]

            # window_left_arr[idx_l, :] = zero_embed[:cfg.pos_embed_size]
            # window_right_arr[idx_r, :] = zero_embed[:cfg.pos_embed_size]

            window_arr_avg = np.stack((window_left_arr, window_right_arr)).mean(axis=0).mean(axis=0)
            embed_rows[ind - start, :] = window_arr_avg
        return embed_rows

    def perform_ko(self, model, pred_data):
        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        indices = self.get_ctcf_indices()
        embed_rows, start = self.convert_df_to_np(pred_data)
        embed_rows = self.ko_indices(embed_rows, start, indices)

        _, ko_pred_df = model.perform_ko(data_loader, embed_rows, start)
        ko_pred_df.to_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        return ko_pred_df

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
        embed_rows, start = self.convert_df_to_np(pred_data)
        embed_rows = self.normalize_embed(embed_rows)

        _, ko_pred_df = model.perform_ko(data_loader, embed_rows, start)
        ko_pred_df.to_csv(cfg.output_directory + "shuffle_%s_norm_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        return ko_pred_df

    def duplicate(self, embed_rows):
        plt_start = 6700
        chunk_start = 6794
        chunk_end = 7008
        dupl_start = 7009
        dupl_end = 7223
        plt_end = 7440

        shift = 215

        # key = "chr" + str(self.chr - 1)
        # cum_pos = int(self.sizes[key])
        # diff = start - (cum_pos + 1)
        lstrow = len(embed_rows) - 1
        startrow = lstrow - plt_end
        endrow = lstrow - dupl_start

        for n in range(startrow, endrow + 1):
            embed_rows[lstrow - n, :] = embed_rows[lstrow - n - shift, :]

        return embed_rows

    def reverse_embeddings(self, embed_rows):
        embed_rows = np.fliplr(embed_rows)
        return embed_rows

    def melo_insert(self, model, pred_data, zero_embed):
        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        embed_rows, start = self.convert_df_to_np(pred_data)
        embed_rows = self.duplicate(embed_rows)
        # embed_rows = self.reverse_embeddings(embed_rows)

        _, melo_pred_df = model.perform_ko(data_loader, embed_rows, zero_embed, start)
        melo_pred_df.to_csv(cfg.output_directory + "combined150_meloafkofusion_chr%s.csv" % str(chr), sep="\t")

        return melo_pred_df


if __name__ == '__main__':

    cfg = config.Config()
    cell = cfg.cell

    # load model
    model_name = "shuffle_" + cell
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    # test_chr = list(range(21, 23))
    test_chr = [22]

    for chr in test_chr:
        print('Testing Start Chromosome: {}'.format(chr))
        pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        # zero_embed = np.load(cfg.output_directory + "combined150_zero_chr%s.npy" % str(chr))
        ko_ob = Knockout(cfg, cell, chr)

        ko_pred_df = ko_ob.perform_ko(model, pred_data)
        # np.save(cfg.output_directory + "ko_predict_chr" + str(chr) + ".npy", ko_pred_df)
        # ko_pred_df = ko_ob.normalize_embed_predict(model, pred_data)
        # melo_pred_df = ko_ob.melo_insert(model, pred_data, zero_embed)

    print("done")
