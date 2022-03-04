import torch
import numpy as np
import pandas as pd
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from training.data_utils import get_samples_sparse, get_cumpos
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from training.config import Config
from analyses.feature_attribution.tf import TFChip
from analyses.plot.plot_utils import get_heatmaps, simple_plot
from training.test_model import test_model

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

    def get_ctcf_indices(self):
        """
        get_ctcf_indices() -> DataFrame
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
        return data

    def convert_df_to_np(self, pred_data, method="hiclstm"):
        """
        convert_df_to_np(pred_data, method) -> DataFrame
        Convert dataframe to np array. Easier to manipulate and provide to torch later.
        Args:
            pred_data (DataFrame): Frame containing get_trained_representations.
            method (string): one of hiclstm, sniper, sca
        """

        "assign start and stop"
        i_start = int(pred_data['i'].min())
        i_stop = int(pred_data['i'].max())
        j_start = int(pred_data['j'].min())
        j_stop = int(pred_data['j'].max())

        if i_start < j_start:
            start = i_start
        else:
            start = j_start

        if i_stop > j_stop:
            stop = i_stop
        else:
            stop = j_stop

        try:
            "try loading representations"
            embed_rows = np.load(
                self.cfg.output_directory + "%s_rep_%s_chr%s.npy" % (method, self.cell, str(self.chr)))
        except:
            "initialize"
            nrows = int(stop - start)
            embed_rows = np.zeros((nrows + 1, self.cfg.pos_embed_size))

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

        "run for all indices"
        for ind in indices:
            "alter according to mode in config"
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

    def compute_kodiff(self, pred_data, ko_pred_df, indices):
        """
        compute_kodiff(pred_data, ko_pred_df, indices) -> Array
        Compute difference between predicted contacts after and before knockout
        Args:
            pred_data (DataFrame): Frame containing predictions before knockout
            ko_pred_df (DataFrame): Frame containing predictions after knockout.
            indices (list): indices that were knocked out
        """

        "initialize"
        indices = np.array(indices)
        diff_list = np.zeros((len(indices), 11))
        win = self.cfg.ko_increment
        diff = np.arange(0, 101, 10)

        "average over indices"
        for i, ind in enumerate(indices):
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
                diff_list[i, j] = mean_diff

        "mean diff over indices"
        mean_diff = np.mean(diff_list, axis=0)
        return mean_diff

    def perform_ko(self, model):
        """
        perform_ko(model) -> Array
        Loads data for chromosome. Loads representations. Alters representations.
        Gets padding representation. Runs through decoder. Computes mean diff between WT and KO.
        Saves predictions.
        Args:
            model (SeqLSTM): model to use for knockout.
        """

        "get knockout indices depending on experiment"
        if self.cfg.ko_experiment == "ctcf":
            if self.cfg.ctcf_indices == "all":
                indices = self.get_ctcf_indices()
            else:
                indices = self.cfg.ctcf_indices_22
        elif self.cfg.ko_experiment == "foxg1":
            indices = [222863]

        "load data"
        data_loader = get_data_loader_chr(self.cfg, self.chr, shuffle=False)

        "get representations"
        representations, start, stop, pred_data = self.get_trained_representations(method="hiclstm")

        "get zero embed"
        self.cfg.full_test = False
        self.cfg.compute_pca = False
        self.cfg.get_zero_pred = True
        zero_embed = test_model(model, self.cfg, chr)

        "alter representations"
        representations, zero_embed = self.ko_representations(representations, start, indices, zero_embed,
                                                              mode=cfg.ko_mode)

        if self.cfg.load_ko:
            ko_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")
        else:
            "run through model using altered representations, save ko predictions"
            _, ko_pred_df = model.perform_ko(data_loader, representations, start, zero_embed, mode="ko")
            ko_pred_df.to_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)),
                              sep="\t")

        "compute difference between WT and KO predictions"
        mean_diff = self.compute_kodiff(pred_data, ko_pred_df, indices)
        return mean_diff, ko_pred_df

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
        optimizer, criterion = model.compile_optimizer()
        data_loader = self.prepare_tal1_lmo2()

        "train and save the model"
        model.train_model(data_loader, criterion, optimizer, writer)
        torch.save(model.state_dict(), cfg.model_dir + cfg.model_name + '.pth')

    def test_tal1_lmo2(self, model):
        """
        test_tal1_lmo2(model) -> No return object
        Test model on 5C data from TAL1 and LMO2 regions.
        Args:
            model (SeqLSTM):  Model to be used to test on 5C data
        """

        "prepare dataloader"
        data_loader = self.prepare_tal1_lmo2()

        "test model"
        _, _, _, pred_df, _ = model.test(data_loader)

        "save predictions"
        pred_df.to_csv(cfg.output_directory + "%s_predictions.csv" % (cfg.cell), sep="\t")
        return pred_df

    def run_tal_experiment(self):
        pred_df = None
        tal1_data = None
        lmo2_data = None
        if cfg.tal_pre:
            "prepare tal1 and lmo2 data"
            self.convert_to_hic_format()

        if cfg.tal_train:
            "train 5C data"
            ko_ob.train_tal1_lmo2(model)

        if cfg.tal_test:
            "test tal1 and lmo2 regions"
            pred_df = ko_ob.test_tal1_lmo2(model)

        if cfg.compare_tal:
            "compare predictions and observed 5C"
            if pred_df is None:
                pred_df = pd.read_csv(cfg.output_directory + "%s_predictions.csv" % (cfg.cell), sep="\t")

            tal1_data = pred_df.loc[pred_df["i"] < 7000]
            lmo2_data = pred_df.loc[pred_df["i"] > 7000]
            tal1_mat, _ = get_heatmaps(tal1_data, no_pred=False)
            lmo2_mat, _ = get_heatmaps(lmo2_data, no_pred=False)

            if cfg.tal_plot_wt:
                simple_plot(tal1_mat)
                simple_plot(lmo2_mat)

        if cfg.check_ko:
            "compare ko and observed 5C"
            if cfg.tal_pre_ko:
                self.cfg.tal_mode = "lmo2_ko"
                self.convert_to_hic_format()
            tal1_ko = pd.read_csv(cfg.output_directory + "tal1_ko.txt", sep="\t")
            lmo2_ko = pd.read_csv(cfg.output_directory + "lmo2_ko.txt", sep="\t")

            tal1_data["pred"] = tal1_ko["v"]
            lmo2_data["pred"] = lmo2_ko["v"]
            tal1_mat, _ = get_heatmaps(tal1_data, no_pred=False)
            lmo2_mat, _ = get_heatmaps(lmo2_data, no_pred=False)

            if cfg.tal_plot_ko:
                simple_plot(tal1_mat)
                simple_plot(lmo2_mat)


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
            mean_diff, ko_pred_df = ko_ob.perform_ko(model)

        if cfg.compare_ko:
            "plot comparison"
            _, _, _, pred_data = ko_ob.get_trained_representations(method="hiclstm")
            if ko_pred_df is None:
                ko_pred_df = pd.read_csv(cfg.output_directory + "hiclstm_%s_afko_chr%s.csv" % (cfg.cell, str(chr)),
                                         sep="\t")
            pred_data = pd.merge(pred_data, ko_pred_df, on=["i", "j"])
            pred_data = pred_data.rename(columns={"ko_pred": "v"})

            hic_mat, st = get_heatmaps(pred_data, no_pred=False)
            simple_plot(hic_mat)
            print("stop")

    "TAL1 and LMO2"
    if cfg.run_tal:
        cfg.cell = "HEK239T"
        cfg.model_name = "shuffle_" + cfg.cell
        ko_ob = Knockout(cfg, cfg.lmo2_chr)
        ko_ob.run_tal_experiment()
