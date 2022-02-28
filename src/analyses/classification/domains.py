import pandas as pd
import os
import numpy as np
from training.config import Config
from analyses.classification.downstream_helper import DownstreamHelper


class Domains:
    """
    Class for getting TAD, subTAD, and boundary data.
    Apply relevant filters.
    Merge non loop domains.
    """
    def __init__(self, cfg, chr, mode="ig"):
        self.rep_data = []
        self.base_name = "_domains.txt"
        self.exp_name = cfg.cell + self.base_name
        self.cell_path = os.path.join(cfg.downstream_dir, "domains", self.exp_name)
        self.cfg = cfg
        self.cell = cfg.cell
        self.chr = chr
        self.chr_tad = 'chr' + str(chr)
        self.mode = mode
        self.down_helper_ob = DownstreamHelper(cfg)
        self.tad_file = cfg.downstream_dir + "/FIREs/" + 'TAD_boundaries.xlsx'

    def get_domain_data(self):
        """
        get_domain_data() -> Dataframe
        Gets Domain data.
        Args:
            NA
        """

        "load domain data"
        data = pd.read_csv(self.cell_path, sep="\s+")
        data = data.loc[data['chr1'] == str(self.chr)].reset_index(drop=True)

        "alters domain data"
        data = self.alter_data(data)
        if self.mode == "ig":
            data.rename(columns={'x1': 'start', 'x2': 'end'},
                        inplace=True)
            data = data.filter(['start', 'end', 'target'], axis=1)
            data["target"] = "Domains"
        return data

    def alter_data(self, data):
        """
        alter_data(data) -> Dataframe
        Convert to resolution. Filter columns.
        Args:
            data (Dataframe): Dataframe containing domain start, end and Target.
        """

        "convert to resolution."
        data["x1"] = (data["x1"]).astype(int) // self.cfg.resolution
        data["x2"] = (data["x2"]).astype(int) // self.cfg.resolution
        data["y1"] = (data["y1"]).astype(int) // self.cfg.resolution
        data["y2"] = (data["y2"]).astype(int) // self.cfg.resolution

        "filter columns"
        data["target"] = pd.Series(np.ones(len(data))).astype(int)
        data = data.filter(['x1', 'x2', 'y1', 'y2', 'target'], axis=1)
        return data

    def get_tad_data(self):
        """
        get_tad_data() -> Dataframe
        Gets TAD data. Converts to resolution. Filters chromosome and columns.
        Args:
            NA
        """

        tads = pd.read_excel(self.tad_file, sheet_name=self.cell, names=["chr", "start", "end"])
        tads = tads.sort_values(by=['start']).reset_index(drop=True)

        "convert to resolution"
        tads["start"] = tads["start"] // self.cfg.resolution
        tads["end"] = tads["end"] // self.cfg.resolution

        tad_data_chr = tads.loc[tads['chr'] == self.chr_tad].reset_index(drop=True)
        tad_data_chr['target'] = 1
        tad_data_chr = tad_data_chr.filter(['start', 'end', 'target'], axis=1)
        if self.mode == "ig":
            tad_data_chr['target'] = "TADs"
        return tad_data_chr

    def augment_tad_negatives(self, tad_df):
        """
        augment_tad_negatives(tad_df) -> Dataframe
        Method to augment TAD negatives. Currently not used.
        Args:
            tad_df (Dataframe): Dataframe containing TAD positions.
        """

        neg_df = pd.DataFrame(columns=['start', 'end', 'target'])
        for i in range(tad_df.shape[0]):
            diff = tad_df.iloc[i]['end'] - tad_df.iloc[i]['start']

            start_neg = tad_df.iloc[i]['start'] - diff
            end_neg = tad_df.iloc[i]['start'] - 1

            if i == 0 or start_neg > tad_df.iloc[i - 1]['end']:
                neg_df = neg_df.append({'start': start_neg, 'end': end_neg, 'target': 0},
                                       ignore_index=True)

        tad_updated = pd.concat([tad_df, neg_df]).reset_index(drop=True)
        return tad_updated

    def merge_domains(self):
        """
        merge_domains() -> Dataframe
        Method to Merge TADs and other contact domains.
        Args:
            NA
        """

        "get domain data"
        domain_data = self.get_domain_data()
        tad_data = self.get_tad_data()

        "merge domain data on positions"
        merged_data = pd.concat([domain_data, tad_data])
        merged_data = merged_data.drop_duplicates(subset=['start', 'end'], keep='last')
        merged_data["target"] = "Merged_Domains"
        return merged_data

    def get_tad_boundaries(self, tf_ob, ctcf="positive"):
        """
        get_tad_boundaries(tf_ob, ctcf) -> Dataframe
        Method to Merge TADs and other contact domains.
        Args:
            tf_ob (TFChip): Object to obtain CTCF data.
            ctcf (string): one of positive, negative, and all
        """

        "converts TAD start and ends to boundary positions."
        tads = self.get_tad_data()
        df_start = tads[["start", "target"]].rename(columns={"start": "pos"})
        df_end = tads[["end", "target"]].rename(columns={"end": "pos"})
        tadbs = pd.concat([df_start, df_end])
        tadbs["target"] = "TADBs"

        "Form CTCF+ or CTCF- TAD boundaries"
        ctcf_data = tf_ob.get_ctcf_data()
        if ctcf == "positive":
            tadbctcf = tadbs[tadbs["pos"].isin(ctcf_data["start"])]
            tadbs = pd.concat([tadbctcf, tadbs[tadbs["pos"].isin(ctcf_data["end"])]])
        elif ctcf == "negative":
            tadbctcf = tadbs[~tadbs["pos"].isin(ctcf_data["start"])]
            tadbs = pd.concat([tadbctcf, tadbs[~tadbs["pos"].isin(ctcf_data["end"])]])

        return tadbs


if __name__ == '__main__':
    chr = 21
    cfg = Config()
    cell = cfg.cell

    rep_ob = Domains(cfg, chr, mode="ig")
    data = rep_ob.get_domain_data()
