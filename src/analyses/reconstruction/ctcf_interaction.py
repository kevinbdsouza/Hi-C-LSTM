import pandas as pd
from training.config import Config
import numpy as np
import torch
from analyses.classification.loops import Loops
from analyses.classification.downstream_helper import DownstreamHelper
from training.data_utils import get_cumpos

pd.options.mode.chained_assignment = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CTCF_Interactions():

    def __init__(self, cfg, chr, mode):
        self.cfg = cfg
        self.mode = mode
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.res = cfg.resolution
        self.genome_len = cfg.genome_len
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.start_end_path = self.cfg.hic_path + self.cfg.start_end_file
        self.start_ends = np.load(self.start_end_path, allow_pickle=True).item()
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode="lstm")

    def get_loop_data(self):
        loop_ob = Loops(cfg, cell, chr)
        loop_data = loop_ob.get_loop_data()
        cum_pos = get_cumpos(self.cfg, self.chr)

        col_list = ['x1', 'x2', 'y1', 'y2']

        loop_data = loop_data.filter(col_list, axis=1)
        loop_data = loop_data.drop_duplicates(keep='first').reset_index(drop=True)
        loop_data[col_list] += cum_pos

        return loop_data


if __name__ == '__main__':
    test_chr = list(range(21, 22))
    cfg = Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    for chr in test_chr:
        ctcf_ob_hic = CTCF_Interactions(cfg, chr, mode='test')
        loop_data = ctcf_ob_hic.get_loop_data()

print("done")
