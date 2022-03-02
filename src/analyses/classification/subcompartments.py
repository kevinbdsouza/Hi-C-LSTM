import pandas as pd
import os
from training.config import Config
from training.data_utils import load_interchrom_hic, get_bin_idx, contactProbabilities
import numpy as np
from analyses.plot.plot_utils import get_heatmaps
from hmmlearn import hmm


class Subcompartments:
    """
    Class for getting subcompartment data.
    Apply relevant filters.
    Obtain subcompartments for GM12878 from Rao.
    to get subcompartments for H1hESC, and HFFhTERT, Create matrix C as given in Rao.
    Rows consist of odd chromosomes and columns consist of even chromosomes.
    Cij is the contacts strength between ith locus in odd chromosome and jth locus in even chromosome.
    Feed this matrix as input to the hmmlearn clusering.
    Do clustering on transpose to cluster even chromosome loci.
    """

    def __init__(self, cfg, chr):
        self.rep_data = []
        self.base_name = "_SC_Rao.bed"
        self.exp_name = cfg.cell + self.base_name
        self.cell_path = os.path.join(cfg.downstream_dir, "subcompartments", self.exp_name)
        self.cfg = cfg
        self.chr = chr
        self.num_subc = 4
        if self.chr % 2 == 0:
            self.even = True

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

    def get_interchrom_data(self):
        """
        get_interchrom_data() -> Dataframe
        Gets interchromosomal data. form odd even matrix for clustering.
        Args:
            NA
        """

        if self.even:
            chr_list = np.arange(1, 23, 2)
        else:
            chr_list = np.arange(2, 23, 2)

        full_data = pd.DataFrame()
        for chry in chr_list:
            data = load_interchrom_hic(self.cfg, self.chr, chry)
            data['i'] = get_bin_idx(np.full(data.shape[0], chr), data['i'], cfg)
            data['j'] = get_bin_idx(np.full(data.shape[0], chr), data['j'], cfg)
            data["v"] = contactProbabilities(data["v"])
            full_data = pd.concat([full_data, data])

        hic_mat, st = get_heatmaps(full_data, no_pred=True)
        return hic_mat

    def hmmlearn(self, hic_mat):
        """
        hmmlearn(hic_mat) -> DataFrame
        sklearn has deprecated gaussian hmm. Therefore use hmmlearn.
        hmmlearn: https://hmmlearn.readthedocs.io/en/stable/.
        Args:
            hic_mat (Array): Array containing Hi-C data.
        """

        cluster_model = hmm.GaussianHMM(n_components=self.num_subc, covariance_type="full", n_iter=100)
        cluster_model.fit(hic_mat)
        state_sequence = cluster_model.predict(hic_mat)
        hic_mat = pd.DataFrame(hic_mat)
        hic_mat["target"] = state_sequence
        return hic_mat


if __name__ == '__main__':
    chr = 21
    cfg = Config()

    rep_ob = Subcompartments(cfg, chr)
    data = rep_ob.get_sc_data()
