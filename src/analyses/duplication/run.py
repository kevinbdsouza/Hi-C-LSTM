from __future__ import division
import torch
import numpy as np
import pandas as pd
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from analyses.knockout.run import Knockout

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'
mode = "test"


class Duplicate():

    def __init__(self, cfg, cell, chr):
        self.cfg = cfg
        self.chr = chr
        self.cell = cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.ko_ob = Knockout(cfg, cell, chr)

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

    def melo_insert(self, model, pred_data):
        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        embed_rows, start, stop = self.ko_ob.convert_df_to_np(pred_data)
        embed_rows = self.duplicate(embed_rows)
        # embed_rows = self.reverse_embeddings(embed_rows)

        _, melo_pred_df = model.perform_ko(data_loader, embed_rows, start, mode="dup")
        melo_pred_df.to_csv(
            cfg.output_directory + "shuffle_%s_meloafkofusion_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

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
        dup_ob = Duplicate(cfg, cell, chr)

        melo_pred_df = dup_ob.melo_insert(model, pred_data)

    print("done")
