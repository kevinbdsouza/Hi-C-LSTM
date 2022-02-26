import pandas as pd
import os
from training.config import Config
from analyses.classification.loops import Loops
from analyses.classification.run import DownstreamTasks


class TFChip:
    def __init__(self, cfg, chr, mode="ig"):
        self.file_path = os.path.join(cfg.downstream_dir, "ctcf")
        self.cohesin_path = os.path.join(cfg.downstream_dir, "cohesin")
        self.file_name = "ENCFF706QLS.bed"
        self.rad21_file_name = "rad21.bed"
        self.smc3_file_name = "smc3.bed"
        self.pol2_file_name = "pol2.bam"
        self.cfg = cfg
        self.mode = mode
        self.chr = 'chr' + str(chr)
        self.downstream_ob = DownstreamTasks(cfg, chr, mode='lstm')

    def get_ctcf_data(self):
        data = pd.read_csv(os.path.join(self.file_path, self.chr + ".bed"), sep="\t", header=None)
        ctcf_data = self.alter_data(data)
        return ctcf_data

    def alter_data(self, data):
        column_list = ["chr", "start", "end", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]
        data.columns = column_list
        data['target'] = "CTCF"

        data["start"] = (data["start"]).astype(int) // self.cfg.resolution
        data["end"] = (data["end"]).astype(int) // self.cfg.resolution
        data = data.filter(['start', 'end', 'target'], axis=1)

        data = data.sort_values('start')
        return data

    def get_cohesin_data(self):
        rad_data = pd.read_csv(os.path.join(self.cohesin_path, self.rad21_file_name), sep="\t", header=None)
        rad_data = rad_data.loc[rad_data[:][0] == self.chr]
        rad_data = self.alter_data(rad_data)

        smc_data = pd.read_csv(os.path.join(self.cohesin_path, self.smc3_file_name), sep="\t", header=None)
        smc_data = smc_data.loc[smc_data[:][0] == self.chr]
        smc_data = self.alter_data(smc_data)

        if self.mode == "ig":
            rad_data["target"] = "RAD21"
            smc_data["target"] = "SMC3"

        return rad_data, smc_data

    def ctcf_in_loops(self, loops="inside"):
        ctcf_data = self.get_ctcf_data()
        rad_data, smc_data = self.get_cohesin_data()
        merged_data = pd.concat([ctcf_data, rad_data, smc_data])
        merged_data = merged_data.drop_duplicates(subset=['start', 'end'], keep='last')
        merged_data["target"] = "CTCF+Cohesin"

        loop_ob = Loops(cfg, chr, mode="ig")
        loop_data = loop_ob.get_loop_data()
        loop_data = loop_data.drop_duplicates(keep='first').reset_index(drop=True)
        loop_data = self.downstream_ob.downstream_helper_ob.get_window_data(loop_data)
        if loops == "inside":
            within_loops = merged_data[merged_data["start"].isin(loop_data["pos"])]
            merged_data = pd.concat([within_loops, merged_data[merged_data["end"].isin(loop_data["pos"])]])
        elif loops == "outside":
            outside_loops = merged_data[~merged_data["start"].isin(loop_data["pos"])]
            merged_data = pd.concat([outside_loops, merged_data[~merged_data["end"].isin(loop_data["pos"])]])

        return merged_data


if __name__ == '__main__':
    chr = 21
    cfg = Config()
    cell = cfg.cell

    rep_ob = TFChip(cfg, cell, mode="ig")
    rad_data, smc_data = rep_ob.get_cohesin_data()
