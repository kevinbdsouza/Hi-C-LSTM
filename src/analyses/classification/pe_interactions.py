import pandas as pd
import re
import os
from training.config import Config


class PeInteractions:
    """
    Class to get Promoters, Enhancers, and interaction data from given cell type.
    Alter it and provide for classification.
    Before using make sure that TSS and enhancer data is loaded for cell type.
    And set to promoter and enhancer starts and ends. If window data is available (GM12878),
    set to window start and end. Use link in manuscript to download data.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.chr = 'chr' + str(chr)
        self.pe_int_path = os.path.join(cfg.downstream_dir, "PE-interactions")
        self.pairs_file = os.path.join(self.pe_int_path, "pairs.csv")

    def get_pe_data(self):
        """
        get_rep_data() -> No return object
        Gets Replication Timing data.
        Args:
            NA
        """

        "load pairs"
        pe_data = pd.read_csv(self.pairs_file, sep=",")
        pe_data = pe_data.sort_values(by=['window_start']).reset_index(drop=True)

        "covert to resolution"
        columns = ["window_start", "window_end", "promoter_start", "promoter_end", "enhancer_start", "enhancer_end"]
        pe_data = pe_data.drop_duplicates(subset="promoter_name", keep='first').reset_index(drop=True)
        pe_data[columns] = pe_data[columns] // self.cfg.resolution

        "filter chromosome"
        pe_data = pe_data.loc[pe_data['window_chrom'] == self.chr].reset_index(drop=True)
        return pe_data

    def get_cell(self, column):
        """
        get_cell(column) -> list
        Gets Cell type.
        Args:
            column (string): Expression involving cell type.
        """
        return re.split(r"\|\s*", column)[0]

    def filter_pe_data(self, pe_data):
        """
        filter_pe_data(pe_data) -> Dataframe
        Filter cell type, and applies changes according to element.
        Args:
            pe_data (Dataframe): Dataframe containing p, e, and window data.
        """

        "filters cell"
        pe_data["cell"] = pe_data["window_name"].apply(self.get_cell)
        pe_data = pe_data.loc[pe_data["cell"] == self.cfg.cell]

        "apply changes according to element"
        if self.cfg.class_element == "Enhancers":
            pe_data = pe_data.filter(['enhancer_start', 'enhancer_end', 'label'], axis=1)
            pe_data.rename(columns={'enhancer_start': 'start', 'enhancer_end': 'end', 'label': 'target'}, inplace=True)
            pe_data = pe_data.assign(target=1)
        elif self.cfg.class_element == "TSS":
            pe_data = pe_data.filter(['promoter_start', 'promoter_end', 'label'], axis=1)
            pe_data.rename(columns={'promoter_start': 'start', 'promoter_end': 'end', 'label': 'target'}, inplace=True)
            pe_data = pe_data.assign(target=1)
        elif self.cfg.class_element == "PE-Interactions":
            pe_data = pe_data.filter(['window_start', 'window_end', 'label'], axis=1)
            pe_data.rename(columns={'window_start': 'start', 'window_end': 'end', 'label': 'target'}, inplace=True)
            pe_data = pe_data.assign(target=1)
        return pe_data


if __name__ == '__main__':
    chr = 21
    cfg = Config()

    pe_ob = PeInteractions(cfg, chr)
