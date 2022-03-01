import pandas as pd
import os
from training.config import Config


class Subcompartments:
    """
    Class for getting subcompartment data.
    Apply relevant filters.
    """

    def __init__(self, cfg, chr):
        self.rep_data = []
        self.base_name = "_SC_Rao.bed"
        self.exp_name = cfg.cell + self.base_name
        self.cell_path = os.path.join(cfg.downstream_dir, "subcompartments", self.exp_name)
        self.cfg = cfg
        self.chr = chr

    def get_sc_data(self):
        """
        get_sc_data() -> Dataframe
        Gets subcompartment data.
        Args:
            NA
        """

        "load data"
        data = pd.read_csv(self.cell_path, sep="\s+", header=None)

        "filter data"
        data = data.loc[:, 0:4]
        data.columns = ["chr", "start", "end", "SC", "target"]
        data = data.loc[data['chr'] == "chr" + str(self.chr)].reset_index(drop=True)

        "convert to resolutions and filter columns"
        data["start"] = (data["start"]).astype(int) // self.cfg.resolution
        data["end"] = (data["end"]).astype(int) // self.cfg.resolution
        data = data.filter(['start', 'end', 'target'], axis=1)
        data = data.replace({'target': {-1: 3, -2: 1, -3: 5, 1: 4}, })
        return data


if __name__ == '__main__':
    chr = 21
    cfg = Config()

    rep_ob = Subcompartments(cfg, chr)
    data = rep_ob.get_sc_data()
