import pandas as pd
import os
from training import config

class TFChip:
    def __init__(self, cfg, cell, chr):
        self.file_path = os.path.join(cfg.downstream_dir, "ctcf")
        self.cohesin_path = os.path.join(cfg.downstream_dir, "cohesin")
        self.file_name = "ENCFF706QLS.bed"
        self.rad21_file_name = "rad21.bed"
        self.smc3_file_name = "smc3.bed"
        self.pol2_file_name = "pol2.bam"
        self.cfg = cfg
        self.chr = chr

    def get_ctcf_data(self):
        data = pd.read_csv(os.path.join(self.file_path, self.chr + ".bed"), sep="\t", header=None)
        ctcf_data = self.alter_data(data)
        return ctcf_data

    def alter_data(self, data):
        column_list = ["chr", "start", "end", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]
        data.columns = column_list
        data['target'] = 1
        ctcf_data = data.filter(['start', 'end', "target"], axis=1)

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

        # pol_data = pd.read_csv(os.path.join(self.cohesin_path, self.pol2_file_name), sep="\t", header=None)
        # pol_data = pol_data.loc[pol_data[:][0] == chr]
        # pol_data = self.alter_data(pol_data)

        return rad_data, smc_data


if __name__ == '__main__':
    data_dir = "/data2/hic_lstm/downstream"

    chr = 21
    cfg = config.Config()
    cell = "GM12878"

    rep_ob = TFChip(cfg, cell, chr)
    # data = rep_ob.get_ctcf_data()

    rad_data, smc_data = rep_ob.get_cohesin_data()
    print("done")
