import pandas as pd
import os
from training.config import Config


class GeneExp:
    """
    Class to get RNA-Seq data from given cell type.
    Alter it and provide for classification.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.gene_info = None
        self.pc_data = None
        self.nc_data = None
        self.rb_data = None
        self.chr = str(chr)
        self.gene_exp_path = os.path.join(cfg.downstream_dir, "RNA-seq")
        self.gene_exp_file = os.path.join(self.gene_exp_path, "Ensembl_v65.Gencode_v10.ENSG.gene_info")
        self.pc_file = os.path.join(self.gene_exp_path, "57epigenomes.RPKM.pc.gz")
        self.nc_file = os.path.join(self.gene_exp_path, "57epigenomes.RPKM.nc.gz")
        self.rb_file = os.path.join(self.gene_exp_path, "57epigenomes.RPKM.rb.gz")
        if cfg.cell == "GM12878":
            self.cell_column = "E116"
        elif cfg.cell == "H1hESC":
            self.cell_column = "E003"
        elif cfg.cell == "HFFhTERT":
            self.cell_column = "E055"

    def get_rna_seq(self):
        """
        get_rna_seq() -> No return object
        Gets RNA-Deq data for PC, NC, and RB modes.
        Args:
            NA
        """
        self.gene_info = pd.read_csv(self.gene_exp_file, sep="\s+", header=None)
        self.gene_info.rename(
            columns={0: 'gene_id', 1: 'chr', 2: 'start', 3: 'end', 4: 'no_idea', 5: 'type', 6: 'gene',
                     7: 'info'}, inplace=True)

        "get protein coding, non protein coding, and reference free modes"
        self.pc_data = pd.read_csv(self.pc_file, compression='gzip', header=0, sep="\s+")
        self.nc_data = pd.read_csv(self.nc_file, compression='gzip', header=0, sep="\s+")
        self.rb_data = pd.read_csv(self.rb_file, compression='gzip', header=0, sep="\s+")

    def filter_rna_seq(self):
        """
        filter_rna_seq() -> Dataframe
        Filters chromosome. Combined modes based on gene id. Converts to resolution.
        Filters cell type.
        Args:
            NA
        """

        "filter chromosome"
        gene_info_chr = self.gene_info.loc[self.gene_info['chr'] == self.chr]

        "combine by gene id"
        self.pc_data = self.pc_data.merge(gene_info_chr, on=['gene_id'], how='inner')
        self.nc_data = self.nc_data.merge(gene_info_chr, on=['gene_id'], how='inner')
        self.rb_data = self.rb_data.merge(gene_info_chr, on=['gene_id'], how='inner')

        rna_seq_chr = pd.concat([self.pc_data, self.nc_data, self.rb_data], ignore_index=True)
        rna_seq_chr = rna_seq_chr.sort_values(by=['start']).reset_index(drop=True)

        "convert to resolution"
        rna_seq_chr["start"] = rna_seq_chr["start"] // self.cfg.resolution
        rna_seq_chr["end"] = rna_seq_chr["end"] // self.cfg.resolution

        "filer by cell"
        rna_seq_chr['target'] = 0
        rna_seq_chr.loc[rna_seq_chr.loc[:, self.cell_column] >= 0.5, 'target'] = 1
        rna_seq_chr = rna_seq_chr.filter(['start', 'end', 'target'], axis=1)
        return rna_seq_chr


if __name__ == '__main__':
    chr = 21
    cfg = Config()
    cell_name = cfg.cell

    rna_seq_ob = GeneExp(cfg, chr)
    rna_seq_ob.get_rna_seq()
