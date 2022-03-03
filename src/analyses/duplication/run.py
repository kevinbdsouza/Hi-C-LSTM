import torch
import numpy as np
import pandas as pd
from training.config import Config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from analyses.knockout.run import Knockout
from training.test_model import test_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Duplicate():
    """
    Class to implement duplication.
    Includes methods that help you duplicate.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.chr = chr
        self.cell = cfg.cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.ko_ob = Knockout(cfg, chr)

    def duplicate(self, embed_rows):
        """
        test_tal1_lmo2(model, cfg) -> No return object
        Args:
            model (SeqLSTM):
            cfg (Config):
        """

        plt_start = 6700
        chunk_start = 6794
        chunk_end = 7008
        dupl_start = 7009
        dupl_end = 7223
        plt_end = 7440

        shift = 215

        lstrow = len(embed_rows) - 1
        startrow = lstrow - plt_end
        endrow = lstrow - dupl_start

        for n in range(startrow, endrow + 1):
            embed_rows[lstrow - n, :] = embed_rows[lstrow - n - shift, :]

        return embed_rows

    def melo_insert(self, model, pred_data):
        """
        test_tal1_lmo2(model, cfg) -> No return object
        Args:
            model (SeqLSTM):
            cfg (Config):
        """

        data_loader, samples = get_data_loader_chr(self.cfg, self.chr)
        embed_rows, start, stop = self.ko_ob.convert_df_to_np(pred_data)
        embed_rows = self.duplicate(embed_rows)
        # embed_rows = self.reverse_embeddings(embed_rows)

        _, melo_pred_df = model.perform_ko(data_loader, embed_rows, start, mode="dup")
        melo_pred_df.to_csv(
            cfg.output_directory + "shuffle_%s_meloafkofusion_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")

        return melo_pred_df


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell

    "load model"
    model_name = cfg.model_name
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    for chr in cfg.dupl_chrs:
        print('Duplication Start Chromosome: {}'.format(chr))

        dup_ob = Duplicate(cfg, chr)

        if cfg.dupl_compute_test:
            "run test if predictions not computed yet"
            test_model(model, cfg, chr)

        pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)),
                                sep="\t")

        "melo insertion"
        melo_pred_df = dup_ob.melo_insert(model, pred_data)
