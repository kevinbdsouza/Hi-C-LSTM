import pandas as pd
import re


class Rep_timing:
    def __init__(self, cfg):

        self.rep_data = []
        self.huvec = "RT_HUVEC_Umbilical"
        self.imr90 = "RT_IMR90_Lung"
        self.k562 = "RT_K562_Bone"
        self.nhek = "RT_NHEK_Keratinocytes_Int92817591_hg38"
        self.gm12878 = "RT_GM12878_Lymphocyte_Int90901931_hg38"
        self.cell_names = ['HUVEC', 'IMR90', 'K562']
        self.cfg = cfg

    def get_rep_data(self, rep_path, cell_names):

        data = None
        for cell in cell_names:

            if cell == "GM12878":
                pass
            elif cell == "HUVEC":
                data = pd.read_csv(rep_path + "/" + cell, sep="\s+", header=None)
                data.columns = ["chr", "start", "end", "rep_timing"]
                data["target"] = 0

                data.loc[data.iloc[:]["rep_timing"] >= 0, 'target'] = 1
                data["start"] = data["start"] // self.cfg.resolution
                data["end"] = data["end"] // self.cfg.resolution

            elif cell == "IMR90":
                data = pd.read_csv(rep_path + "/" + cell, sep="\s+", header=None)
                data.columns = ["chr", "start", "end", "rep_timing"]

                data.loc[data.iloc[:]["rep_timing"] >= 0, 'target'] = 1
                data["start"] = data["start"] // self.cfg.resolution
                data["end"] = data["end"] // self.cfg.resolution

            elif cell == "K562":
                data = pd.read_csv(rep_path + "/" + cell, sep="\s+", header=None)
                data.columns = ["chr", "start", "end", "rep_timing"]

                data.loc[data.iloc[:]["rep_timing"] >= 0, 'target'] = 1
                data["start"] = data["start"] // self.cfg.resolution
                data["end"] = data["end"] // self.cfg.resolution

            elif cell == "NHEK":
                pass

            self.rep_data.append(data)

    def filter_rep_data(self, chrom_rep):

        filtered_data = []
        for i, cell in enumerate(self.rep_data):
            data = self.rep_data[i]
            new_data = data.loc[data['chr'] == chrom_rep].reset_index(drop=True)
            filtered_data.append(new_data)

        return filtered_data


if __name__ == '__main__':
    rep_timing_path = "/data2/latent/data/downstream/replication_timing"

    chrom_rep = 21

    rep_ob = Rep_timing()
    rep_ob.get_rep_data(rep_timing_path, rep_ob.cell_names)
    fire_labeled = rep_ob.filter_rep_data(chrom_rep)

    print("done")
