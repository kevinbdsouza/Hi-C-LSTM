import pandas as pd
import re


class Fires:
    def __init__(self, cfg, chr, mode="ig"):
        self.fire_data = None
        self.tad_data = None
        self.cfg = cfg
        self.chr = chr
        self.mode = mode
        self.path = cfg.downstream_dir + "/FIREs/"
        self.fire_file = self.path + "fires.pkl"
        self.cell = "GM12878"

    def get_fire_data(self):
        fires = pd.read_pickle(self.fire_file)

        "convert to resolution"
        fires["start"] = fires["start"] // self.cfg.resolution
        fires["end"] = fires["end"] // self.cfg.resolution

        fire_chosen = fires.iloc[:, 0:10]

        self.fire_data = fire_chosen

    def filter_fire_data(self):
        fire_data_chr = self.fire_data.loc[self.fire_data['chr'] == self.chr].reset_index(drop=True)

        fire_data_chr['target'] = 0
        fire_data_chr.loc[fire_data_chr['GM12878'] >= 0.5, 'target'] = 1
        fire_labeled = fire_data_chr.filter(['start', 'end', 'target'], axis=1)
        if self.mode == "ig":
            fire_labeled = fire_labeled.loc[fire_labeled["target"] == 1]
            fire_labeled["target"] = "FIREs"
        return fire_labeled


if __name__ == '__main__':
    fire_path = "/opt/data/latent/data/downstream/FIREs"

    chrom_fire = 21
    chrom_tad = 'chr21'
    cell_names = ['GM12878', 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']

    fire_ob = Fires()
    fire_ob.get_fire_data(fire_path)
    fire_labeled = fire_ob.filter_fire_data(chrom_fire)

    fire_ob.get_tad_data(fire_path, cell_names)
    tad_filtered = fire_ob.filter_tad_data(chrom_tad)

    print("done")
