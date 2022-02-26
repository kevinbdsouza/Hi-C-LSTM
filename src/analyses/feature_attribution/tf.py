import pandas as pd
import os
from training.config import Config
from analyses.classification.loops import Loops


class TFChip:
    def __init__(self, cfg, chr, mode="ig"):
        self.ctcf_path = os.path.join(cfg.downstream_dir, "ctcf")
        self.cohesin_path = os.path.join(cfg.downstream_dir, "cohesin")
        self.tf_path = os.path.join(cfg.downstream_dir, "tfs")
        self.ctcf_file_name = "ENCFF706QLS.bed"
        self.rad21_file_name = "rad21.bed"
        self.smc3_file_name = "smc3.bed"
        self.tf_bed_name = "ENCFF957SRJ.bed"
        self.cfg = cfg
        self.mode = mode
        self.loop_chr = chr
        self.chr = 'chr' + str(chr)

    def get_chip_data(self):
        """
        get_chip_data() -> DataFrame
        Gets CHipSeq positions. Filters columns. Converts them to desired resolution.
        Args:
            NA
        """
        column_list = ["chr", "start_full", "end_full", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]

        chip_data = pd.read_csv(os.path.join(self.tf_path, self.tf_bed_name), sep="\t", header=None)
        chip_data = chip_data.loc[chip_data[:][0] == self.chr]
        chip_data.columns = column_list

        chip_data["start"] = (chip_data["start_full"]).astype(int) // self.cfg.resolution
        chip_data["end"] = (chip_data["end_full"]).astype(int) // self.cfg.resolution
        chip_data = chip_data.filter(['start', 'end', 'start_full', 'end_full', 'target'], axis=1)

        chip_data = chip_data.sort_values('start')
        return chip_data

    def get_ctcf_data(self):
        """
        get_ctcf_data() -> DataFrame
        Gets CTCF CHipSeq positions. Filters columns. Converts them to desired resolution.
        Args:
            NA
        """

        data = pd.read_csv(os.path.join(self.ctcf_path, self.chr + ".bed"), sep="\t", header=None)
        ctcf_data = self.alter_data(data)
        return ctcf_data

    def alter_data(self, data):
        """
        alter_data(data) -> DataFrame
        Filters columns. Converts them to desired resolution.
        Args:
            data (DataFrame): The loaded CTCF positions.
        """

        column_list = ["chr", "start", "end", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]
        data.columns = column_list
        data['target'] = "CTCF"

        data["start"] = (data["start"]).astype(int) // self.cfg.resolution
        data["end"] = (data["end"]).astype(int) // self.cfg.resolution
        data = data.filter(['start', 'end', 'target'], axis=1)

        data = data.sort_values('start')
        return data

    def get_cohesin_data(self):
        """
        get_ctcf_data() -> DataFrame, DataFrame
        Gets RAD21 and SMC3 CHipSeq positions. Filters columns. Converts them to desired resolution.
        Args:
            NA
        """
        rad_data = pd.read_csv(os.path.join(self.cohesin_path, self.rad21_file_name), sep="\t", header=None)
        rad_data = rad_data.loc[rad_data[:][0] == self.chr]
        rad_data = self.alter_data(rad_data)
        rad_data["target"] = "RAD21"

        smc_data = pd.read_csv(os.path.join(self.cohesin_path, self.smc3_file_name), sep="\t", header=None)
        smc_data = smc_data.loc[smc_data[:][0] == self.chr]
        smc_data = self.alter_data(smc_data)
        smc_data["target"] = "SMC3"

        return rad_data, smc_data

    def ctcf_in_loops(self, loops="inside"):
        """
        ctcf_in_loops(loops) -> DataFrame
        Gets common CTCF and Cohesin positions. Obtains Loop data.
        If inside is specified, gets common positions inside loops.
        If outside is specified, gets common positions outside loops.
        Args:
            loops (string): one of all, inside, and outside
        """

        ctcf_data = self.get_ctcf_data()
        rad_data, smc_data = self.get_cohesin_data()
        merged_data = pd.concat([ctcf_data, rad_data, smc_data])
        merged_data = merged_data.drop_duplicates(subset=['start', 'end'], keep='last')
        merged_data["target"] = "CTCF+Cohesin"

        loop_ob = Loops(self.cfg, self.loop_chr, mode="ig")
        loop_data = loop_ob.get_loop_data()
        loop_data = loop_data.drop_duplicates(keep='first').reset_index(drop=True)
        loop_data = loop_ob.down_helper_ob.get_window_data(loop_data)
        if loops == "inside":
            within_loops = merged_data[merged_data["start"].isin(loop_data["pos"])]
            merged_data = pd.concat([within_loops, merged_data[merged_data["end"].isin(loop_data["pos"])]])
            merged_data["target"] = "CTCF+Cohesin_Loop"
        elif loops == "outside":
            outside_loops = merged_data[~merged_data["start"].isin(loop_data["pos"])]
            merged_data = pd.concat([outside_loops, merged_data[~merged_data["end"].isin(loop_data["pos"])]])
            merged_data["target"] = "CTCF+Cohesin_NonLoop"

        return merged_data


if __name__ == '__main__':
    chr = 21
    cfg = Config()
    cell = cfg.cell

    rep_ob = TFChip(cfg, cell, mode="ig")
    rad_data, smc_data = rep_ob.get_cohesin_data()
