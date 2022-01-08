import pandas as pd
from training.config import Config
import numpy as np
import torch
from analyses.classification.loops import Loops
from analyses.classification.downstream_helper import DownstreamHelper
from training.data_utils import get_cumpos
from analyses.plot.plot_utils import simple_plot

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
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode="other")

    def get_loop_data(self):
        loop_ob = Loops(cfg, cell, chr)
        loop_data = loop_ob.get_loop_data()
        cum_pos = get_cumpos(self.cfg, self.chr)

        col_list = ['x1', 'x2', 'y1', 'y2']

        loop_data = loop_data.filter(col_list, axis=1)
        loop_data = loop_data.drop_duplicates(keep='first').reset_index(drop=True)
        loop_data[col_list] += cum_pos

        return loop_data

    def plot_loops(self, pred_data, loop_data):
        st = int(pred_data["i"].min())
        pred_data["i"] = pred_data["i"] - st
        pred_data["j"] = pred_data["j"] - st
        loop_data["x1"] = loop_data["x1"] - st
        loop_data["x2"] = loop_data["x2"] - st
        loop_data["y1"] = loop_data["y1"] - st
        loop_data["y2"] = loop_data["y2"] - st
        nr = int(pred_data["j"].max()) + 1
        rows = np.array(pred_data["i"]).astype(int)
        cols = np.array(pred_data["j"]).astype(int)

        hic_mat = np.zeros((nr, nr))
        hic_mat[rows, cols] = np.array(pred_data["v"])
        hic_upper = np.triu(hic_mat)
        hic_mat[cols, rows] = np.array(pred_data["pred"])
        hic_lower = np.tril(hic_mat)
        hic_mat = hic_upper + hic_lower
        hic_mat[np.diag_indices_from(hic_mat)] /= 2
        for i in range(len(loop_data)):
            hic_win = hic_mat[loop_data.iloc[i, "x1"]:loop_data.iloc[i, "x2"],
                      loop_data.iloc[i, "y1"]:loop_data.iloc[i, "y2"]]
            simple_plot(hic_win)
            print("here")
        # hic_win = hic_mat[6701:7440, 6701:7440]
        # hic_win = hic_mat[900:1450, 900:1450]
        pass


if __name__ == '__main__':
    test_chr = list(range(21, 22))
    cfg = Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    for chr in test_chr:
        pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        ctcf_ob_hic = CTCF_Interactions(cfg, chr, mode='test')
        loop_data = ctcf_ob_hic.get_loop_data()

        ctcf_ob_hic.plot_loops(pred_data, loop_data)

print("done")
