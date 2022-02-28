import pandas as pd
import os
import numpy as np
from training.config import Config
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
        """
        get_loop_data() -> Dataframe
        Gets Loop data. Filter columns. Set Target.
        Args:
            NA
        """

        "load loop data"
        data = pd.read_csv(self.cell_path, sep="\s+")
        data = data.loc[data['chr1'] == str(self.chr)].reset_index(drop=True)

        "alter loop data"
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
        """
        alter_data(data) -> Dataframe
        Convert to resolution and filter columns.
        Args:
            data (Dataframe): Dataframe containing positions and target.
        """

        "convert to resolution"
        cols = ["x1", "x2", "y1", "y2"]
        data[cols] = (data[cols]).astype(int) // self.cfg.resolution

        "filter columns"
        data["target"] = pd.Series(np.ones(len(data))).astype(int)
        data = data.filter(['x1', 'x2', 'y1', 'y2', 'target'], axis=1)
        return data


if __name__ == '__main__':
    chr = 21
    cfg = Config()

    rep_ob = Loops(cfg, chr, mode="class")
    data = rep_ob.get_loop_data()
