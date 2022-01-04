import pandas as pd
from training.config import Config
import numpy as np
import traceback
import torch

pd.options.mode.chained_assignment = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HiC_R2():

    def __init__(self, cfg, chr, mode):
        self.cfg = cfg
        self.mode = mode
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.res = cfg.resolution
        self.genome_len = cfg.genome_len
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.start_end_path = self.cfg.hic_path + self.cfg.start_end_file
        self.start_ends = np.load(self.start_end_path, allow_pickle=True).item()

    def hic_r2(self, hic_predictions):
        hic_predictions.columns = ["i", "j", "v", "pred"] + list(np.arange(2 * self.cfg.pos_embed_size))
        hic_predictions["diff"] = np.abs(hic_predictions["i"] - hic_predictions["j"]).astype(int)
        hic_data = hic_predictions.sort_values(by=['i']).reset_index(drop=True)

        r2_frame = pd.DataFrame(columns=["diff", "r2"])

        start = self.start_ends["chr" + str(self.chr)]["start"] + self.get_cumpos()
        stop = self.start_ends["chr" + str(self.chr)]["stop"] + self.get_cumpos()
        for d in range(0, stop - start):

            try:
                subset_hic = hic_data.loc[hic_data["diff"] == d]
                if len(subset_hic) == 0:
                    continue

                og_hic = subset_hic["v"]
                predicted_hic = subset_hic["pred"]

                r2 = self.find_r2(og_hic, predicted_hic)

                if not np.isfinite(r2):
                    continue
                r2_frame = r2_frame.append({"diff": d, "r2": r2}, ignore_index=True)

            except Exception as e:
                print(traceback.format_exc())
                continue

        return r2_frame

    def find_r2(self, og_hic, predicted_hic):

        mean_og = og_hic.mean()
        ss_tot = ((og_hic.sub(mean_og)) ** 2).sum()
        ss_res = ((og_hic - predicted_hic) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)

        return r2

    def get_cumpos(self):
        chr_num = self.chr
        if chr_num == 1:
            cum_pos = 0
        else:
            key = "chr" + str(chr_num - 1)
            cum_pos = self.sizes[key]

        return cum_pos


if __name__ == '__main__':
    test_chr = list(range(15, 23))
    cfg = Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    for chr in test_chr:
        data_ob_hic = HiC_R2(cfg, chr, mode='test')
        hic_predictions = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)),
                                      sep="\t")
        hic_predictions = hic_predictions.drop(['Unnamed: 0'], axis=1)
        r2_frame = data_ob_hic.hic_r2(hic_predictions)
        r2_frame.to_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")

print("done")
