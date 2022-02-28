import pandas as pd
from training.config import Config


class Fires:
    """
    Class to get FIRE data.
    Filter according to chromsome and cell type.
    """

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
        """
        get_fire_data() -> Dataframe
        Gets FIRE data. Converts to resolution.
        Args:
            NA
        """
        fires = pd.read_pickle(self.fire_file)

        "convert to resolution"
        fires["start"] = fires["start"] // self.cfg.resolution
        fires["end"] = fires["end"] // self.cfg.resolution

        fire_chosen = fires.iloc[:, 0:10]
        self.fire_data = fire_chosen

    def filter_fire_data(self):
        """
        filter_fire_data() -> Dataframe
        Filter according to chromosome and cell type. Set target.
        Args:
            NA
        """

        "filter chromosome"
        fire_data_chr = self.fire_data.loc[self.fire_data['chr'] == self.chr].reset_index(drop=True)

        fire_data_chr['target'] = 0
        fire_data_chr.loc[fire_data_chr['GM12878'] >= 0.5, 'target'] = 1
        fire_labeled = fire_data_chr.filter(['start', 'end', 'target'], axis=1)

        "for IG"
        if self.mode == "ig":
            fire_labeled = fire_labeled.loc[fire_labeled["target"] == 1]
            fire_labeled["target"] = "FIREs"
        return fire_labeled


if __name__ == '__main__':
    chr = 21
    cfg = Config
    fire_ob = Fires(cfg, chr, mode="ig")
