import torch
import numpy as np
import pandas as pd
from training.config import Config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from analyses.knockout.run import Knockout
from training.test_model import test_model
from analyses.plot.plot_utils import get_heatmaps, simple_plot
from analyses.reconstruction.hic_r2 import HiC_R2

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
        self.r2_ob = HiC_R2(cfg, chr)

    def duplicate(self, representations):
        """
        duplicate(representations) -> Array
        Representations to duplicate in the specified region.
        Args:
            representations (Array): Representations to duplicate.
        """

        "positions"
        lstrow = len(representations) - 1
        dupl_start = cfg.dupl_start_rep
        plt_end = cfg.plt_end
        shift = cfg.melo_shift
        startrow = lstrow - plt_end
        endrow = lstrow - dupl_start

        "shift representations"
        for n in range(startrow, endrow + 1):
            representations[lstrow - n, :] = representations[lstrow - n - shift, :]

        return representations

    def melo_insert(self, model):
        """
        melo_insert(modelx`) -> No return object
        Perform melo insertion. Saves resulting predictions. Specify fusion if duplicating with fusion.
        Args:
            model (SeqLSTM): Model to be used for duplication.
        """

        "load data for chromosome"
        data_loader = get_data_loader_chr(cfg, chr, shuffle=False)

        "get representations"
        representations, start, stop, pred_data = self.ko_ob.get_trained_representations(method="hiclstm")

        "get zero embed"
        self.cfg.full_test = False
        self.cfg.compute_pca = False
        self.cfg.get_zero_pred = True
        zero_embed = test_model(model, self.cfg, chr)

        "duplicate"
        representations = self.duplicate(representations)

        "load duplicated"
        if self.cfg.dupl_load_data:
            if self.cfg.dupl_mode == "fusion":
                melo_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_meloafkofusion_chr%s.csv" % (cell, str(chr)),
                                           sep="\t")
            else:
                melo_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_meloafko_chr%s.csv" % (cell, str(chr)),
                                       sep="\t")
        else:
            "perform duplication"
            if self.cfg.dupl_mode == "fusion":
                _, melo_pred_df = model.perform_ko(data_loader, representations, start, zero_embed, mode="fusion")
                melo_pred_df.to_csv(
                    cfg.output_directory + "hiclstm_%s_meloafkofusion_chr%s.csv" % (cell, str(chr)), sep="\t")
            else:
                _, melo_pred_df = model.perform_ko(data_loader, representations, start, zero_embed, mode="dup")
                melo_pred_df.to_csv(
                    cfg.output_directory + "hiclstm_%s_meloafko_chr%s.csv" % (cell, str(chr)), sep="\t")

        return melo_pred_df


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell

    "load model"
    model_name = cfg.model_name
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    melo_pred_df = None
    for chr in cfg.dupl_chrs:
        print('Duplication Start Chromosome: {}'.format(chr))

        dup_ob = Duplicate(cfg, chr)

        if cfg.dupl_compute_test:
            "run test if predictions not computed yet"
            cfg.compute_pca = False
            cfg.get_zero_pred = False
            cfg.full_test = True
            test_model(model, cfg, chr)

        if cfg.melo_insert:
            "melo insertion"
            melo_pred_df = dup_ob.melo_insert(model)

        if cfg.compare_dup:
            "plot comparison"
            pred_data = dup_ob.r2_ob.get_prediction_df(method="hiclstm", decoder="full")

            if melo_pred_df is None:
                if cfg.dupl_mode == "fusion":
                    melo_pred_df = pd.read_csv(
                        cfg.output_directory + "hiclstm_%s_meloafkofusion_chr%s.csv" % (cell, str(chr)),
                        sep="\t")
                else:
                    melo_pred_df = pd.read_csv(
                        cfg.output_directory + "hiclstm_%s_meloafko_chr%s.csv" % (cell, str(chr)),
                        sep="\t")

            pred_data = pd.merge(pred_data, melo_pred_df, on=["i", "j"])
            pred_data = pred_data.rename(columns={"ko_pred": "v"})

            hic_mat, st = get_heatmaps(pred_data, no_pred=False)
            simple_plot(hic_mat[cfg.plt_start:cfg.plt_end, cfg.plt_start:cfg.plt_end], mode="reds")
            print("done")
