import pandas as pd
from training.config import Config
import numpy as np
import traceback
import torch
from training.data_utils import get_data_loader_chr
from analyses.knockout.run import Knockout
from training.decoder import Decoder
import time
from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HiC_R2():
    """
    Class to compute R2 along different distances along the genomic axis.
    Includes methods that help you do this.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.cell = cfg.cell
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.res = cfg.resolution
        self.genome_len = cfg.genome_len
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.start_end_path = self.cfg.hic_path + self.cfg.start_end_file
        self.start_ends = np.load(self.start_end_path, allow_pickle=True).item()

    def hic_r2(self, hic_predictions):
        """
        hic_r2(hic_predictions) -> DataFrame
        Computes R2 for different distances along the genomic axis.
        Args:
            hic_predictions (DataFrame): The dataframe with embeddings and position IDs.
        """

        "make diff column"
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

                "compute r2"
                r2 = self.find_r2(og_hic, predicted_hic)

                if not np.isfinite(r2):
                    continue
                r2_frame = r2_frame.append({"diff": d, "r2": r2}, ignore_index=True)

            except Exception as e:
                print(traceback.format_exc())
                continue

        return r2_frame

    def find_r2(self, og_hic, predicted_hic):
        """
        find_r2(og_hic, predicted_hic) -> Series
        Computes R2 according to observed and predicted arguments
        Args:
            og_hic (Series): Series with observed Hi-C at a particular difference distance
            predicted_hic (Series): Series with predicted Hi-C at a particular difference distance
        """

        mean_og = og_hic.mean()
        ss_tot = ((og_hic.sub(mean_og)) ** 2).sum()
        ss_res = ((og_hic - predicted_hic) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)

        return r2

    def get_cumpos(self):
        """
        get_cumpos() -> int
        Returns cumulative index upto the end of the previous chromosome.
        Args:
            NA
        """

        chr_num = self.chr
        if chr_num == 1:
            cum_pos = 0
        else:
            key = "chr" + str(chr_num - 1)
            cum_pos = self.sizes[key]

        return cum_pos

    def get_prediction_df(self, cfg, chr, method="hiclstm", decoder="lstm"):
        if method == "hiclstm":
            pred_data = pd.read_csv(cfg.output_directory + "%s_%s_%s_chr%s.csv" % (cfg.cell, method, decoder, str(chr)),
                                    sep="\t")
            pred_data = pred_data.drop(['Unnamed: 0'], axis=1)
        elif method == "sniper":
            pred_data = None, None
        elif method == "sca":
            pred_data = None, None

        return pred_data

    def get_trained_representations(self, method="hiclstm"):
        ko_ob = Knockout(self.cfg, self.cell, self.chr)

        if method == "hiclstm":
            pred_data = pd.read_csv(
                self.cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (self.cell, str(self.chr)),
                sep="\t")
            pred_data = pred_data.drop(['Unnamed: 0'], axis=1)
            representations, start, stop = ko_ob.convert_df_to_np(pred_data, method="hiclstm")
        elif method == "sniper":
            "obtain from sniper"
            representations, start = None, None
        elif method == "sca":
            "obtain from sca"
            representations, start = None, None

        return representations, start

    def run_decoders(self, representations, cfg, chr, start, decoder="lstm"):
        """
        run_decoders(representations, cfg, chr, start, decoder) -> No return object
        Obtains data for given chr and specified cell in config.
        Trains the passed representations from method through the decoder of choice.
        Works on one chromosome at a time.
        Args:
            representations (Array): Representation matrix
            cfg (Config): The configuration to use for the experiment.
            chr (int): The chromosome to test.
            start (int): Start indiex in data to offset during training
            decoder (string): one of lstm, cnn, and fc
        """

        "Set up Tensorboard logging"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter('./tensorboard_logs/' + cfg.decoder_name + timestr)

        "Initalize decoder and load decoder weights if they exist"
        decoder_ob = Decoder(cfg, device).to(device)
        decoder_ob.load_weights()

        "Initalize optimizer"
        optimizer, criterion = decoder_ob.compile_optimizer()

        "get data"
        data_loader = get_data_loader_chr(cfg, chr)

        "train decoder"
        decoder_ob.train_decoders(data_loader, representations, start, criterion, optimizer, writer, decoder=decoder)

    def test_decoders(self, representations, cfg, chr, start, method="hiclstm", decoder="lstm"):
        """
        test_decoders(representations, cfg, chr, start, method, decoder) -> No return object
        Loads loads data, tests the model, and saves the predictions in a csv file.
        Works on one chromosome at a time.
        Args:
            representations (array): The representations from the method
            cfg (Config): The configuration to use for the experiment.
            chr (int): The chromosome to test.
            start (int): Start indiex in data to offset during training
            method (string): one of hiclstm, sniper, sca
            decoder (string): one of lstm, cnn, and fc
        """

        "Initalize decoder and load decoder weights if they exist"
        decoder_ob = Decoder(cfg, device).to(device)
        decoder_ob.load_weights()

        "get data"
        data_loader = get_data_loader_chr(cfg, chr)

        "train decoder"
        predictions, pred_df = decoder_ob.test_decoder(data_loader, representations, start, decoder=decoder)

        "save predictions"
        pred_df.to_csv(cfg.output_directory + "%s_%s_%s_chr%s.csv" % (cfg.cell, method, decoder, str(chr)), sep="\t")


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name
    test_chr = cfg.chr_test_list

    for chr in test_chr:
        r2_ob_hic = HiC_R2(cfg, chr, mode='test')
        hic_predictions = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)),
                                      sep="\t")
        hic_predictions = hic_predictions.drop(['Unnamed: 0'], axis=1)
        r2_frame = r2_ob_hic.hic_r2(hic_predictions)
        r2_frame.to_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")
