import pandas as pd
import re


class PeInteractions:
    def __init__(self, cfg):
        self.pe_data = None
        self.cfg = cfg

    def get_pe_data(self, pe_int_path):
        pe_pairs = pd.read_csv(pe_int_path + '/pairs.csv', sep=",")
        pe_pairs = pe_pairs.sort_values(by=['window_start']).reset_index(drop=True)

        ''' decimate by 25 to get the positions at 25 bp resolution '''
        pe_pairs = pe_pairs.drop_duplicates(subset="promoter_name", keep='first').reset_index(drop=True)
        pe_pairs["window_start"] = pe_pairs["window_start"] // self.cfg.resolution
        pe_pairs["window_end"] = pe_pairs["window_end"] // self.cfg.resolution
        pe_pairs["promoter_start"] = pe_pairs["promoter_start"] // self.cfg.resolution
        pe_pairs["promoter_end"] = pe_pairs["promoter_end"] // self.cfg.resolution
        pe_pairs["enhancer_start"] = pe_pairs["enhancer_start"] // self.cfg.resolution
        pe_pairs["enhancer_end"] = pe_pairs["enhancer_end"] // self.cfg.resolution
        pe_pairs["cell"] = None

        self.pe_data = pe_pairs

    def filter_pe_data(self, chrom):
        pe_data_chr = self.pe_data.loc[self.pe_data['window_chrom'] == chrom].reset_index(drop=True)

        for i in range(len(pe_data_chr)):
            cell_line = re.split(r"\|\s*", pe_data_chr.iloc[i]["window_name"])[0]
            if cell_line == "K562":
                pe_data_chr.loc[i, "cell"] = 'E123'
            elif cell_line == "HeLa-S3":
                pe_data_chr.loc[i, "cell"] = 'E117'
            elif cell_line == "GM12878":
                pe_data_chr.loc[i, "cell"] = 'E116'
            elif cell_line == "IMR90":
                pe_data_chr.loc[i, "cell"] = 'E017'

        return pe_data_chr


if __name__ == '__main__':
    rna_seq_path = "/opt/data/latent/data/downstream/RNA-seq"
    pe_int_path = "/opt/data/latent/data/downstream/PE-interactions"

    chromosome = 'chr21'
    cell_name = 'E003'

    pe_ob = PeInteractions()
    pe_ob.get_pe_data(pe_int_path)
    print("done")
