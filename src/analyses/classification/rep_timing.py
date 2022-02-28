import pandas as pd
from training.config import Config
import os


class Rep_timing:
    """
    Class to get Replication Timing data from given cell type.
    Alter it and provide for classification.
    """

    def __init__(self, cfg, chr):
        self.rep_path = os.path.join(cfg.downstream_dir, "replication_timing")
        self.full_path = os.path.join(self.rep_path, "GM12878.bedgraph")
        self.cfg = cfg
        self.cell = cfg.cell
        self.chr = 'chr' + str(chr)

    def get_rep_data(self):
        """
        get_rep_data() -> Dataframe
        Gets Replication Timing data.
        Args:
            NA
        """

        "load file"
        data = pd.read_csv(self.full_path, sep="\s+", header=None)
        data.columns = ["chr", "start", "end", "rep_timing"]
        data["target"] = 0

        "set target and convert to resolution "
        data.loc[data.iloc[:]["rep_timing"] >= 0, 'target'] = 1
        data["start"] = data["start"] // self.cfg.resolution
        data["end"] = data["end"] // self.cfg.resolution

        "filter chr"
        data = data.loc[data['chr'] == self.chr].reset_index(drop=True)

        "filter columns"
        data = data.filter(['start', 'end', 'target'], axis=1)
        return data


if __name__ == '__main__':
    chr = 21
    cfg = Config()
    rep_ob = Rep_timing(cfg, chr)
