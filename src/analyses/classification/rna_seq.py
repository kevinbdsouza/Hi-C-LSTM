import pandas as pd


class RnaSeq:
    def __init__(self, cfg):
        self.gene_info = None
        self.pc_data = None
        self.nc_data = None
        self.rb_data = None
        self.cfg = cfg  

    def get_rna_seq(self, rna_seq_path):
        self.gene_info = pd.read_csv(rna_seq_path + '/Ensembl_v65.Gencode_v10.ENSG.gene_info', sep="\s+",
                                     header=None)
        self.gene_info.rename(
            columns={0: 'gene_id', 1: 'chr', 2: 'start', 3: 'end', 4: 'no_idea', 5: 'type', 6: 'gene',
                     7: 'info'}, inplace=True)

        self.pc_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.pc.gz", compression='gzip', header=0,
                                   sep="\s+",
                                   error_bad_lines=False)

        self.nc_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.nc.gz", compression='gzip', header=0,
                                   sep="\s+",
                                   error_bad_lines=False)

        self.rb_data = pd.read_csv(rna_seq_path + "/57epigenomes.RPKM.rb.gz", compression='gzip', header=0,
                                   sep="\s+",
                                   error_bad_lines=False)

    def filter_rna_seq(self, chrom):
        gene_info_chr = self.gene_info.loc[self.gene_info['chr'] == chrom]

        pc_data_chr = self.pc_data.merge(gene_info_chr, on=['gene_id'], how='inner')

        nc_data_chr = self.nc_data.merge(gene_info_chr, on=['gene_id'], how='inner')

        rb_data_chr = self.rb_data.merge(gene_info_chr, on=['gene_id'], how='inner')

        rna_seq_chr = pd.concat([pc_data_chr, nc_data_chr, rb_data_chr], ignore_index=True)
        rna_seq_chr = rna_seq_chr.sort_values(by=['start']).reset_index(drop=True)

        ''' decimate by 25 to get the positions at 25 bp resolution '''
        rna_seq_chr["start"] = rna_seq_chr["start"] // self.cfg.resolution
        rna_seq_chr["end"] = rna_seq_chr["end"] // self.cfg.resolution

        return rna_seq_chr


if __name__ == '__main__':
    rna_seq_path = "/opt/data/latent/data/downstream/RNA-seq"
    pe_int_path = "/opt/data/latent/data/downstream/PE-interactions"

    chromosome = '21'
    cell_name = 'E003'
    cfg = None

    rna_seq_ob = RnaSeq(cfg)
    rna_seq_ob.get_rna_seq(rna_seq_path)
    print("done")
