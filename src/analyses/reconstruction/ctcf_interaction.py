import pandas as pd
from training.config import Config
import numpy as np
import torch
from analyses.classification.downstream_helper import DownstreamHelper
from training.data_utils import get_cumpos
from analyses.plot.plot_utils import simple_plot, get_heatmaps
from analyses.classification.domains import Domains

pd.options.mode.chained_assignment = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CTCF_Interactions():
    """
    Class to plot compute interaction dots.
    Includes methods that help you do this.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.res = cfg.resolution
        self.genome_len = cfg.genome_len
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.start_end_path = self.cfg.hic_path + self.cfg.start_end_file
        self.start_ends = np.load(self.start_end_path, allow_pickle=True).item()
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode="other")

    def plot_domain_edges(self, pred_data):
        """
        plot_domain_edges(pred_data) -> No return object
        Gets Loop data from Loops file.
        Args:
            NA
        """

        "converts prediction to heatmaps"
        hic_mat, st = get_heatmaps(pred_data)

        "gets domain data"
        dom_ob = Domains(self.cfg, self.cfg.cell, self.chr)
        dom_data = dom_ob.get_domain_data()

        "computes windows at domain edges"
        th = self.cfg.ctcf_dots_threshold
        mean_map_og = np.zeros((2 * th, 2 * th))
        mean_map_pred = np.zeros((2 * th, 2 * th))
        num = 0
        for n in range(len(dom_data)):
            x1 = dom_data.loc[n]["x1"] - st + get_cumpos(self.cfg, chr)
            x2 = dom_data.loc[n]["x2"] - st + get_cumpos(self.cfg, chr)
            y1 = dom_data.loc[n]["y1"] - st + get_cumpos(self.cfg, chr)
            y2 = dom_data.loc[n]["y2"] - st + get_cumpos(self.cfg, chr)

            if (x2 - x1) <= th - 1:
                continue
            else:
                num += 1
                hic_win_og = hic_mat[x1 - th:x1 + th, y2 - th:y2 + th]
                hic_win_pred = hic_mat[x2 - th:x2 + th, y1 - th:y1 + th]
                mean_map_og = mean_map_og + hic_win_og
                mean_map_pred = mean_map_pred + hic_win_pred

        "plot domain edges"
        mean_map_og = mean_map_og / num
        mean_map_pred = mean_map_pred / num
        simple_plot(mean_map_og)
        simple_plot(mean_map_pred.T)


if __name__ == '__main__':
    cfg = Config()
    test_chr = cfg.chr_test_list
    cell = cfg.cell
    model_name = cfg.model_name

    for chr in test_chr:
        pred_data = pd.read_csv(cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")
        ctcf_ob_hic = CTCF_Interactions(cfg, chr)
        ctcf_ob_hic.plot_domain_edges(pred_data)
