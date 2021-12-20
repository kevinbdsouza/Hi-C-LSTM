import pandas as pd
import re
import os
import numpy as np
from train_fns import config
from train_fns.data_prep_hic import DataPrepHic
from downstream.downstream_helper import DownstreamHelper


class Subcompartments:
    def __init__(self, cfg, cell, chr, mode):
        self.rep_data = []
        if mode == "Rao":
            self.base_name = "_SC_Rao.bed"
        elif mode == "Sniper":
            self.base_name = "_SC_Sniper.bed"
        self.exp_name = cell + self.base_name
        self.cell_path = os.path.join(cfg.downstream_dir, "subcompartments", self.exp_name)
        self.cfg = cfg
        self.chr = chr
        self.data_ob = DataPrepHic(cfg, cell, mode='test', chr=str(chr))
        self.down_helper_ob = DownstreamHelper(cfg, cell, chr, mode="test")

    def get_sc_data(self):
        data = pd.read_csv(self.cell_path, sep="\s+", header=None)
        data = data.loc[:, 0:4]
        data.columns = ["chr", "start", "end", "SC", "target"]
        data = data.loc[data['chr'] == "chr" + str(self.chr)].reset_index(drop=True)

        data = self.alter_data(data)
        return data

    def alter_data(self, data):
        data["start"] = (data["start"]).astype(int) // self.cfg.resolution
        data["end"] = (data["end"]).astype(int) // self.cfg.resolution
        data = data.filter(['start', 'end', 'target'], axis=1)

        return data


if __name__ == '__main__':
    data_dir = "/data2/hic_lstm/downstream"

    chr = 21
    cfg = config.Config()
    cell = "GM12878"

    rep_ob = Subcompartments(cfg, cell, chr, mode="Rao")
    data = rep_ob.get_sc_data()

    print("done")
