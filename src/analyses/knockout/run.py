import torch
import numpy as np
import pandas as pd
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from training.data_utils import get_samples_sparse, get_cumpos, contactProbabilities
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from training.config import Config
from analyses.feature_attribution.tf import TFChip
from analyses.classification.domains import Domains
from analyses.plot.plot_utils import get_heatmaps, simple_plot, indices_diff_mat
from training.test_model import test_model
from random import sample

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Knockout():
    """
    Class to perform knockout experiments.
    Includes methods that help you do knockout.
    """

    def __init__(self, cfg, chr):
        self.cfg = cfg
        self.chr = chr
        self.cell = cfg.cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path, allow_pickle=True).item()
        self.hek_file = cfg.hic_path + cfg.cell + "/HEK239T-WT.matrix"
        self.tal1ko_file = cfg.hic_path + cfg.cell + "/Tal1KO.matrix"
        self.lmo2ko_file = cfg.hic_path + cfg.cell + "/Lmo2KO.matrix"

    def get_trained_representations(self, method="hiclstm"):
        """
        get_trained_representations(method) -> Array, int
        Gets fully trained representations for given method, cell type and chromosome.
        obtain sniper and sca representations from respective methods.
        Should contain SNIPER and SCA positions end internal representations.
        Args:
            method (string): one of hiclstm, sniper, sca
        """

        pred_data = pd.read_csv(
            self.cfg.output_directory + "%s_%s_predictions_chr%s.csv" % (method, self.cell, str(self.chr)),
            sep="\t")
        pred_data = pred_data.drop(['Unnamed: 0'], axis=1)
        representations, start, stop = self.convert_df_to_np(pred_data, method=method)
        pred_data = pred_data.filter(['i', 'j', 'v', 'pred'], axis=1)
        return representations, start, stop, pred_data

    def get_tadbs(self):
        """
        get_tadbs() -> Array
        Gets TAD Boundaries to knockout.
        Args:
            NA
        """

        dom_ob = Domains(cfg, chr, mode="ko")
        tf_ob = TFChip(cfg, chr)
        tadbs = dom_ob.get_tad_boundaries(tf_ob, ctcf="negative")
        cum_pos = get_cumpos(self.cfg, self.chr)
        tadbs = tadbs + cum_pos
        tadbs = np.array(tadbs)
        return tadbs

    def get_ctcf_indices(self):
        """
        get_ctcf_indices() -> Array
        Gets CTCF positions to knockout.
        Args:
            NA
        """

        "gets CTCF positions"
        ctcf_ob = TFChip(cfg, chr)
        data = ctcf_ob.get_ctcf_data()
        data = data.filter(['start'], axis=1)

        "converts to cumulative indices"
        cum_pos = get_cumpos(self.cfg, self.chr)
        data["start"] = data["start"] + cum_pos
        indices = np.array(data["start"])
        return indices

    def convert_df_to_np(self, pred_data, method="hiclstm"):
        """
        convert_df_to_np(pred_data, method) -> DataFrame
        Convert dataframe to np array. Easier to manipulate and provide to torch later.
        Args:
            pred_data (DataFrame): Frame containing get_trained_representations.
            method (string): one of hiclstm, sniper, sca
        """

        "assign start and stop"
        start = min(int(pred_data['i'].min()), int(pred_data['j'].min()))
        stop = max(int(pred_data['i'].max()), int(pred_data['j'].max()))

        try:
            "try loading representations"
            embed_rows = np.load(
                self.cfg.output_directory + "%s_rep_%s_chr%s.npy" % (method, self.cell, str(self.chr)))
        except:
            "initialize"
            nrows = int(stop - start) + 1
            embed_rows = np.zeros((nrows, self.cfg.pos_embed_size))

            i_old = 0
            j_old = 0
            for r in range(len(pred_data)):
                i_new = int(pred_data.loc[r, "i"])

                "skip already seen positions"
                if i_new == i_old:
                    continue
                else:
                    "assign representations"
                    i_old = i_new
                    if np.all((embed_rows[i_new - start, :] == 0)):
                        col = list(np.arange(self.cfg.pos_embed_size))
                        col = [str(x) for x in col]
                        embed_rows[i_new - start, :] = np.array(pred_data.loc[r, col])

                "repeat for j"
                j_new = int(pred_data.loc[r, "j"])
                if j_new == j_old:
                    continue
                else:
                    "assign representations"
                    j_old = j_new
                    if np.all((embed_rows[j_new - start, :] == 0)):
                        col = list(np.arange(self.cfg.pos_embed_size, 2 * self.cfg.pos_embed_size))
                        col = [str(x) for x in col]
                        embed_rows[j_new - start, :] = np.array(pred_data.loc[r, col])

            np.save(self.cfg.output_directory + "%s_rep_%s_chr%s.npy" % (method, self.cell, str(self.chr)), embed_rows)
        return embed_rows, start, stop

    def normalize_embed(self, representations, zero_embed):
        """
        normalize_embed(representations, zero_embed) -> Array, Array
        Normalize each row separately.
        Args:
            representations (Array): Array of representations to normalize.
            zero_embed (Array): pading representation
        """

        "normalize representations"
        for n in range(len(representations)):
            norm = np.linalg.norm(representations[n, :])
            if norm == 0:
                continue
            else:
                representations[n, :] = representations[n, :] / norm

        "normalize padding"
        norm = np.linalg.norm(zero_embed)
        zero_embed = zero_embed / norm
        return representations, zero_embed

    def ko_representations(self, representations, start, indices, zero_embed, mode="average"):
        """
        ko_representations(representations, start, indices, zero_embed, mode) -> Array, Array
        Alter representations to feed to knockout.
        Args:
            representations (Array): Array of representations.
            start (ind): start indice in chromosome.
            indices (list): indices to knockout
            zero_embed (Array): Padding representation
            mode (string): one of average, zero, padding, shift, normalize, or reverse
        """

        window = self.cfg.ko_window
        size = len(representations)

        if isinstance(indices, int):
            indices = [indices]

        "alter according to mode in config"
        for ind in indices:
            if mode == "average":
                if ind - start - window < 0 or ind - start + window > size:
                    window = int(window // 2)

                window_left_arr = representations[ind - start - window: ind - start, :].copy()
                window_right_arr = representations[ind - start + 1: ind - start + window + 1, :].copy()

                window_arr_avg = np.stack((window_left_arr, window_right_arr)).mean(axis=0).mean(axis=0)
                representations[ind - start, :] = window_arr_avg
            elif mode == "zero":
                representations[ind - start, :] = np.zeros((1, cfg.pos_embed_size))
            elif mode == "shift":
                representations[ind - start:size - 1, :] = representations[ind - start + 1:size, :]
                representations[size - 1, :] = np.zeros((1, cfg.pos_embed_size))
            elif mode == "padding":
                representations[ind - start, :] = zero_embed[:cfg.pos_embed_size]

        if mode == "reverse":
            representations = np.fliplr(representations)
            zero_embed = np.flip(zero_embed)
        elif mode == "normalize":
            representations, zero_embed = self.normalize_embed(representations, zero_embed)

        return representations, zero_embed

    def compute_kodiff(self, pred_data, ko_pred_df, ind):
        """
        compute_kodiff(pred_data, ko_pred_df, indices) -> Array
        Compute difference between predicted contacts after and before knockout
        Args:
            pred_data (DataFrame): Frame containing predictions before knockout
            ko_pred_df (DataFrame): Frame containing predictions after knockout.
            indices (list): indices that were knocked out
        """

        "initialize"
        ko_diffs = np.zeros((11,))
        win = self.cfg.ko_increment
        diff = np.arange(0, 101, 10)

        "compute diff"
        for j, d in enumerate(diff):
            "take subset of knockout data in window"
            if j == 0:
                subset_og = pred_data.loc[pred_data["i"] == ind]
            else:
                subset_og = pred_data.loc[
                    ((pred_data["i"] <= ind + j * win) & (pred_data["i"] > ind + (j - 1) * win))
                    | ((pred_data["i"] >= ind - j * win) & (pred_data["i"] < ind - (j - 1) * win))]
            if subset_og.empty:
                continue

            "take subset of original data in window"
            if j == 0:
                subset_ko = ko_pred_df.loc[ko_pred_df["i"] == ind]
            else:
                subset_ko = ko_pred_df.loc[
                    ((ko_pred_df["i"] <= ind + j * win) & (ko_pred_df["i"] > ind + (j - 1) * win))
                    | ((ko_pred_df["i"] >= ind - j * win) & (ko_pred_df["i"] < ind - (j - 1) * win))]

            "compute mean diff in window"
            merged_df = pd.merge(subset_og, subset_ko, on=["i", "j"])
            merged_df = merged_df.filter(['i', 'j', 'pred', 'ko_pred'], axis=1)
            mean_diff = np.mean(np.array(merged_df["ko_pred"]) - np.array(merged_df["pred"]))
            ko_diffs[j] = mean_diff
        return ko_diffs

    def perform_ko(self, model):
        """
        perform_ko(model) -> Array
        Loads data for chromosome. Loads representations. Alters representations.
        Gets padding representation. Runs through decoder. Computes mean diff between WT and KO.
        Saves predictions.
        Args:
            model (SeqLSTM): model to use for knockout.
        """

        cfg = self.cfg

        "load data"
        if cfg.run_tal and cfg.hnisz_region == "tal1":
            self.cfg.get_tal1_only = True
            data_loader = self.prepare_tal1_lmo2()
        elif cfg.run_tal and cfg.hnisz_region == "lmo2":
            self.cfg.get_lmo2_only = True
            data_loader = self.prepare_tal1_lmo2()
        else:
            data_loader = get_data_loader_chr(cfg, self.chr, shuffle=False)

        "get zero embed"
        cfg.full_test = False
        cfg.compute_pca = False
        cfg.get_zero_pred = True
        zero_embed = test_model(model, cfg, self.chr)

        "get knockout indices depending on experiment"
        if cfg.run_tal:
            if cfg.hnisz_region == "tal1":
                cfg.ko_experiment = "ctcf"
                indices = cfg.tal1ko_indices
            elif cfg.hnisz_region == "lmo2":
                cfg.ko_experiment = "ctcf"
                indices = cfg.lmo2ko_indices + get_cumpos(cfg, 11)
        else:
            if cfg.ko_experiment == "ctcf":
                if cfg.ctcf_indices == "all":
                    indices = ko_ob.get_ctcf_indices()
                    indices = sample(list(indices), 100)
                else:
                    indices = ko_ob.cfg.ctcf_indices_22
            elif cfg.ko_experiment == "foxg1":
                indices = cfg.foxg1_indices
            elif cfg.ko_experiment == "tadbs":
                indices = ko_ob.get_tadbs()

        "plotting and metrics"
        n_indices = len(indices)
        diff_list = np.zeros((n_indices, 11))
        diff_mat = np.zeros((n_indices, 200, 200))
        "run for all indices"
        for i, indice in enumerate(indices):
            "get representations"
            representations, start, stop, pred_data = self.get_trained_representations(method="hiclstm")

            "alter representations"
            representations, zero_embed = self.ko_representations(representations, start, indice, zero_embed,
                                                                  mode=cfg.ko_mode)

            if self.cfg.load_ko:
                ko_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)),
                                         sep="\t")
            else:
                "run through model using altered representations, save ko predictions"
                _, ko_pred_df = model.perform_ko(data_loader, representations, start, zero_embed, mode="ko")
                if self.cfg.save_kopred:
                    ko_pred_df.to_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)),
                                      sep="\t")

            "compute difference between WT and KO predictions"
            if self.cfg.compute_avg_diff:
                ko_diffs = self.compute_kodiff(pred_data, ko_pred_df, indice)
                diff_list[i] = ko_diffs

            "get merged heatmap"
            pred_data = pd.merge(pred_data, ko_pred_df, on=["i", "j"])
            pred_data = pred_data.rename(columns={"ko_pred": "v"})
            hic_mat, st = get_heatmaps(pred_data, no_pred=False)
            simple_plot(hic_mat, mode="reds")

            "get diff mat"
            hic_win = indices_diff_mat(indice, st, hic_mat, mode=cfg.ko_experiment)
            n_win = len(hic_win)
            diff_mat[i, :n_win, :n_win] = hic_win

        diff_mat = diff_mat.mean(axis=0)
        ko = np.triu(diff_mat)
        pred = np.tril(diff_mat).T
        diff_mat = ko - pred
        simple_plot(diff_mat, mode="diff")
        np.save(cfg.output_directory + "tad_diff_zero_ctctn.npy", diff_mat)
        mean_diff = np.mean(diff_list, axis=1)
        return mean_diff, ko_pred_df, pred_data

    def change_index(self, list_split):
        """
        change_index(list_split) -> list, list
        get locations from index.
        Args:
            list_split (list): list
        """

        "format index"
        temp = [k.split('|')[-1] for k in list_split]
        chr_list = []
        index_list = []
        for t in temp:
            index = t.split(':')
            chr_list.append(index[0])
            index_list.append(index[1].split('-'))

        "prepare locations list"
        loc_list = []
        for ind in index_list:
            loc = int(((int(ind[0]) + int(ind[1])) / 2) // 10000)
            loc_list.append(loc)

        return loc_list, chr_list

    def convert_to_hic_format(self):
        """
        convert_to_hic_format() -> No return object.
        Assigns positions and chr. Convert 5C to Hi-C like format.
        Args:
            NA.
        """

        if self.cfg.tal_mode == "wt":
            hek_mat = pd.read_csv(self.hek_file, sep="\t")
        elif self.cfg.tal_mode == "tal1_ko":
            hek_mat = pd.read_csv(self.tal1ko_file, sep="\t")
        elif self.cfg.tal_mode == "lmo2_ko":
            hek_mat = pd.read_csv(self.lmo2ko_file, sep="\t")

        "get positions"
        index, chr_list = self.change_index(list(hek_mat.index))
        columns, _ = self.change_index(hek_mat.columns)

        "assign rows, columns and chr"
        hek_mat.index = index
        hek_mat.columns = columns
        hek_mat["chr"] = chr_list

        "get matrices for TAL1 and LMO2"
        tal1_mat = hek_mat.loc[hek_mat["chr"] == "chr1"]
        tal1_mat = tal1_mat.iloc[:, 0:285]
        lmo2_mat = hek_mat.loc[hek_mat["chr"] == "chr11"]
        lmo2_mat = lmo2_mat.iloc[:, 286:632]
        tal1_mat = tal1_mat.groupby(level=0, axis=1).sum()
        tal1_mat = tal1_mat.groupby(level=0, axis=0).sum()
        lmo2_mat = lmo2_mat.groupby(level=0, axis=1).sum()
        lmo2_mat = lmo2_mat.groupby(level=0, axis=0).sum()

        "prepare data in the form of Hi-C"
        tal_i = list(tal1_mat.index)
        tal_j = tal1_mat.columns
        lmo2_i = list(lmo2_mat.index)
        lmo2_j = lmo2_mat.columns

        tal_df = pd.DataFrame(columns=["i", "j", "v"])
        for i in tal_i:
            for j in tal_j:
                tal_df = tal_df.append({"i": i, "j": j, "v": tal1_mat.loc[i][j]}, ignore_index=True)

        lmo2_df = pd.DataFrame(columns=["i", "j", "v"])
        for i in lmo2_i:
            for j in lmo2_j:
                lmo2_df = lmo2_df.append({"i": i, "j": j, "v": lmo2_mat.loc[i][j]}, ignore_index=True)

        "save data"
        if self.cfg.tal_mode == "wt":
            tal_df.to_csv(cfg.hic_path + cfg.cell + "/tal_df.txt", sep="\t")
            lmo2_df.to_csv(cfg.hic_path + cfg.cell + "/lmo2_df.txt", sep="\t")
        else:
            tal_df.to_csv(cfg.output_directory + "tal1_ko.txt", sep="\t")
            lmo2_df.to_csv(cfg.output_directory + "lmo2_ko.txt", sep="\t")

    def prepare_tal1_lmo2(self):
        """
        prepare_tal1_lmo2(cfg) -> DataLoader
        prepare dataloader to train.
        Args:
            cfg (Config): config to be used to run the experiment.
        """

        "load Hi-C like data"
        tal_df = pd.read_csv(cfg.hic_path + cfg.cell + "/tal_df.txt", sep="\t")
        lmo2_df = pd.read_csv(cfg.hic_path + cfg.cell + "/lmo2_df.txt", sep="\t")

        "preprocess"
        tal_df = tal_df.drop(['Unnamed: 0'], axis=1)
        lmo2_df = lmo2_df.drop(['Unnamed: 0'], axis=1)
        tal_df[['i', 'j']] = tal_df[['i', 'j']].astype('int64')
        lmo2_df[['i', 'j']] = lmo2_df[['i', 'j']].astype('int64')

        "prepare indices and values for TAL1 in chromosome 1"
        values = torch.empty(0, cfg.sequence_length)
        input_idx = torch.empty(0, cfg.sequence_length, 2)
        input_idx_tal1, values_tal1 = get_samples_sparse(tal_df, 1, cfg)
        values_tal1 = F.pad(input=values_tal1, pad=(0, 4, 0, 0), mode='constant', value=0)
        input_idx_tal1 = F.pad(input=input_idx_tal1, pad=(0, 0, 0, 4, 0, 0), mode='constant', value=0)
        values = torch.cat((values, values_tal1.float()), 0)
        input_idx = torch.cat((input_idx, input_idx_tal1), 0)

        if self.cfg.get_tal1_only:
            "create tal dataloader"
            dataset = torch.utils.data.TensorDataset(input_idx, values)
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)
            return data_loader

        if self.cfg.get_lmo2_only:
            values = torch.empty(0, cfg.sequence_length)
            input_idx = torch.empty(0, cfg.sequence_length, 2)

        "prepare indices and values for LMO2 in chromosome 11"
        input_idx_lmo2, values_lmo2 = get_samples_sparse(lmo2_df, 11, cfg)
        values = torch.cat((values, values_lmo2.float()), 0)
        input_idx = torch.cat((input_idx, input_idx_lmo2), 0)

        "create dataloader"
        dataset = torch.utils.data.TensorDataset(input_idx, values)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

        return data_loader

    def train_tal1_lmo2(self, model):
        """
        train_tal1_lmo2(model, cfg) -> No return object
        Train model on 5C data from TAL1 and LMO2 regions.
        Args:
            model (SeqLSTM): Model to be used to train on 5C data
        """

        "summary writer"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter('./tensorboard_logs/' + cfg.model_name + timestr)

        "initialize optimizer and prepare dataloader"
        self.cfg.get_tal1_only = False
        self.cfg.get_lmo2_only = False
        optimizer, criterion = model.compile_optimizer()
        data_loader = self.prepare_tal1_lmo2()

        "train and save the model"
        model.train_model(data_loader, criterion, optimizer, writer)
        torch.save(model.state_dict(), cfg.model_dir + cfg.model_name + '.pth')

    def test_tal1_lmo2(self, model):
        """
        test_tal1_lmo2(model) -> DataFrame
        Test model on 5C data from TAL1 and LMO2 regions.
        Args:
            model (SeqLSTM):  Model to be used to test on 5C data
        """

        "prepare dataloader"
        data_loader = self.prepare_tal1_lmo2()

        "test model"
        self.cfg.full_test = True
        self.cfg.compute_pca = False
        self.cfg.get_zero_pred = False
        _, _, _, pred_df, _ = model.test(data_loader)

        "save predictions"
        pred_df.to_csv(self.cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (self.cell, str(self.chr)),
                       sep="\t")
        return pred_df

    def perform_tal1_ko(self, model):
        """
        perform_tal1_ko(model) -> DataFrame
        Performs knockout of selected sites in TAL1 and LMO2 regions.
        Args:
            model (SeqLSTM):  Model to be used to test on 5C data
        """

        "save representations"
        self.chr = 1
        self.cfg.get_tal1_only = True
        ko_ob.test_tal1_lmo2(model)

        "perform ko"
        self.cfg.hnisz_region = "tal1"
        _, ko_pred_df, _ = self.perform_ko(model)
        return ko_pred_df

    def perform_lmo2_ko(self, model):
        """
        perform_tal1_ko(model) -> DataFrame
        Performs knockout of selected sites in TAL1 and LMO2 regions.
        Args:
            model (SeqLSTM):  Model to be used to test on 5C data
        """

        ko_pred_df = None

        "save representations"
        self.chr = 11
        self.cfg.get_lmo2_only = True
        ko_ob.test_tal1_lmo2(model)

        return ko_pred_df

    def run_tal_experiment(self):
        pred_df = None
        ko_pred_df = None
        tal1_data = None
        lmo2_data = None

        "load model"
        model = SeqLSTM(self.cfg, device).to(device)
        model.load_weights()

        if cfg.tal_pre:
            "prepare tal1 and lmo2 data"
            self.convert_to_hic_format()

        if cfg.tal_train:
            "train 5C data"
            ko_ob.train_tal1_lmo2(model)

        if cfg.tal_test:
            "test tal1 and lmo2 regions"
            ko_ob.cfg.get_tal1_only = False
            ko_ob.cfg.get_lmo2_only = False
            pred_df = ko_ob.test_tal1_lmo2(model)

        if cfg.perform_tal1_ko:
            ko_pred_df = ko_ob.perform_tal1_ko(model)

        if cfg.perform_lmo2_ko:
            ko_pred_df = ko_ob.perform_lmo2_ko(model)

        if cfg.compare_tal:
            "compare predictions and observed 5C"
            if pred_df is None:
                pred_df = pd.read_csv(cfg.output_directory + "%s_predictions.csv" % (cfg.cell), sep="\t")

            tal1_data = pred_df.loc[pred_df["i"] < 7000]
            lmo2_data = pred_df.loc[pred_df["i"] > 7000]
            tal1_mat, _ = get_heatmaps(tal1_data.copy(), no_pred=False)
            lmo2_mat, _ = get_heatmaps(lmo2_data.copy(), no_pred=False)

            if cfg.tal_plot_wt:
                simple_plot(tal1_mat, mode="reds")
                simple_plot(lmo2_mat, mode="reds")

        if cfg.check_ko:
            "compare ko and observed 5C"
            if cfg.tal_pre_ko:
                self.cfg.tal_mode = "tal1_ko"
                self.convert_to_hic_format()

                tal1_ko = pd.read_csv(cfg.output_directory + "tal1_ko.txt", sep="\t")
                lmo2_ko = pd.read_csv(cfg.output_directory + "lmo2_ko.txt", sep="\t")

                tal1_data["pred"] = contactProbabilities(tal1_ko["v"])
                lmo2_data["pred"] = contactProbabilities(lmo2_ko["v"])
                tal1_mat, _ = get_heatmaps(tal1_data, no_pred=False)
                lmo2_mat, _ = get_heatmaps(lmo2_data, no_pred=False)

                if cfg.tal_plot_ko:
                    simple_plot(tal1_mat, mode="reds")
                    simple_plot(lmo2_mat, mode="reds")

            tal1_wt = np.load(cfg.output_directory + "tal1_wt.npy")
            tal1_ogko = np.load(cfg.output_directory + "tal1_comp.npy")
            tal1_predko = np.load(cfg.output_directory + "tal1_comp_predko.npy")
            print("stop")


if __name__ == '__main__':
    cfg = Config()

    "load model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    if cfg.ko_experiment == "foxg1":
        cfg.chr_test_list = cfg.foxg1_chr

    ko_pred_df = None
    for chr in cfg.chr_test_list:
        print('Knockout Start Chromosome: {}'.format(chr))
        ko_ob = Knockout(cfg, chr)

        if cfg.ko_compute_test:
            "run test if predictions not computed yet"
            test_model(model, cfg, chr)

        if cfg.perform_ko:
            "perform ko"
            ko_ob.cfg.run_tal = False
            mean_diff, ko_pred_df, pred_data = ko_ob.perform_ko(model)

        if cfg.compare_ko:
            "plot comparison"
            if ko_pred_df is None:
                _, _, _, pred_data = ko_ob.get_trained_representations(method="hiclstm")
                ko_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)),
                                         sep="\t")
            pred_data = pd.merge(pred_data, ko_pred_df, on=["i", "j"])
            pred_data = pred_data.rename(columns={"ko_pred": "v"})

            hic_mat, st = get_heatmaps(pred_data, no_pred=False)
            simple_plot(hic_mat, mode="reds")
            print("done")

    "TAL1 and LMO2"
    if cfg.run_tal:
        cfg.cell = "HEK239T"
        cfg.model_name = "shuffle_" + cfg.cell
        ko_ob = Knockout(cfg, cfg.lmo2_chr)
        ko_ob.run_tal_experiment()
