import pandas as pd
import os
import numpy as np
from training import config
from analyses.classification.downstream_helper import DownstreamHelper


class Loops:
    """
    Class to get Loop Domain data.
    Apply Relevant filters.
    """

    def __init__(self, cfg, chr, mode="ig"):
        self.rep_data = []
        self.base_name = "_loops_motifs.txt"
        self.exp_name = cfg.cell + self.base_name
        self.cell_path = os.path.join(cfg.downstream_dir, "loops", self.exp_name)
        self.cfg = cfg
        self.chr = chr
        self.mode = mode
        self.down_helper_ob = DownstreamHelper(cfg)

    def get_loop_data(self):
        data = pd.read_csv(self.cell_path, sep="\s+")
        data = data.loc[data['chr1'] == str(self.chr)].reset_index(drop=True)

        data = self.alter_data(data)

        pos_matrix = pd.DataFrame()
        for i in range(2):
            if i == 0:
                temp_data = data.rename(columns={'x1': 'start', 'x2': 'end'},
                                        inplace=False)
            else:
                temp_data = data.rename(columns={'y1': 'start', 'y2': 'end'},
                                        inplace=False)

            temp_data = temp_data.filter(['start', 'end', 'target'], axis=1)
            pos_matrix = pos_matrix.append(temp_data)

        if self.mode == "ig":
            pos_matrix["target"] = "Loops"

        return pos_matrix

    def alter_data(self, data):
        data["x1"] = (data["x1"]).astype(int) // self.cfg.resolution
        data["x2"] = (data["x2"]).astype(int) // self.cfg.resolution
        data["y1"] = (data["y1"]).astype(int) // self.cfg.resolution
        data["y2"] = (data["y2"]).astype(int) // self.cfg.resolution

        data["target"] = pd.Series(np.ones(len(data))).astype(int)
        data = data.filter(['x1', 'x2', 'y1', 'y2', 'target'], axis=1)

        return data


if __name__ == '__main__':
    data_dir = "/data2/hic_lstm/downstream"

    chr = 21
    cfg = config.Config()
    cell = "GM12878"

    rep_ob = Loops(cfg, cell, chr)
    data = rep_ob.get_loop_data()

    print("done")
