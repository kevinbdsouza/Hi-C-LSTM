import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from training.config import Config


class PlotFns:
    """
    Class to plot major analyses plots. Uses plot utils.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.output_directory

    def plot_stackedbar(self, df_main, tasks, colors):
        """
        plot_stackedbar(df_main, tasks, colors) -> No return object
        Gets upper and lower comperison heatmaps.
        Args:
            df_main (DataFrame): Frame with methods and metric values
            tasks (List): List of tasks
            colors (List): List of colors
        """

        df_main = df_main.T
        df_main.columns = df_main.iloc[0]
        df_main = df_main.drop(["Tasks"], axis=0)
        fields = df_main.columns.tolist()

        "figure and axis"
        fig, ax = plt.subplots(1, figsize=(22, 12))

        "plot bars"
        left = len(df_main) * [0]
        for idx, name in enumerate(fields):
            plt.barh(df_main.index, df_main[name], left=left, color=colors[idx])
            left = left + df_main[name]

        "legend"
        plt.rcParams.update({'font.size': 22})
        plt.legend(tasks, bbox_to_anchor=([0.02, 1, 0, 0]), ncol=6, frameon=False, fontsize=14)

        "remove spines"
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        "format x ticks"
        xticks = np.arange(0, 36.1, 4)
        xlabels = ['{}'.format(i) for i in np.arange(0, 36.1, 4)]
        plt.xticks(xticks, xlabels, fontsize=20)

        "adjust limits and draw grid lines"
        plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("Cumulative Prediction Score", fontsize=20)
        plt.savefig("/home/kevindsouza/Downloads/H1hESC_metrics.png")

    def plot_main(self, cell, metric, df_columns, df_lists, xlabel, ylabel, colors, markers, labels, form_df=True,
                  adjust=False,
                  save=True):
        """
        plot_main(cell, metric, df_columns, df_lists, adjust=False, save=True) -> No return object
        Main plotting function
        Args:
            cell (string): one of GM12878, HFFhTERT, H1hESC, WTC11
            metric (string): one of map, auroc, accuracy or fscore
            df_columns (List): list of df columns
            df_lists (List): list containing data for columns
            xlabel (string): xlabel
            ylabel (string): ylabel
            colors (List): list of colors
            markers (List): list of markers
            labels (List): list of labels
            adjust (bool): If true adjusts bottom
            save (bool): if true saves figure
        """

        if form_df:
            df_main = pd.DataFrame(columns=df_columns)
            for i, v in enumerate(df_columns):
                df_main[v] = df_lists[i]
        else:
            df_main = pd.read_csv(self.path + "%s_%s_df.csv" % (cell, metric), sep="\t")
            df_main = df_main.drop(['Unnamed: 0'], axis=1)

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        for i, l in enumerate(labels):
            plt.plot(df_columns[0], df_columns[i + 1], data=df_main, marker=markers[i], markersize=16, color=colors[i],
                     linewidth=3,
                     label=l)

        plt.legend(fontsize=18)

        if adjust:
            plt.subplots_adjust(bottom=0.35)

        plt.show()

        if save:
            plt.savefig("/home/kevindsouza/Downloads/x.png")

    def plot_combined(self, cell, metric, ylabel):
        """
        plot_combined(cell, metric, ylabel) -> No return object
        Plots given metrics in the given cell
        Args:
            cell (string): One of GM12878, H1hESC, HFFhTERT
            metric (string): One of map, auroc, accuracy, fscore
        """

        xlabel = "Prediction Target"
        ylabel = ylabel
        colors = ["C3", "C0", "C1", "C2", "C4", "C5"]
        markers = ['o', '*', 'X', '^', 'D', 's']
        labels = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        df_columns = ["Tasks"] + labels

        "plot"
        self.plot_main(cell, metric, df_columns, None, xlabel, ylabel, colors, markers, labels, form_df=False, adjust=True,
                       save=False)

    def plot_map_celltypes(self):
        """
        plot_mAP_celltypes() -> No return object
        Plots mAP in all celltypes
        Args:
            NA
        """

        tasks = ["Gene Expression", "Enhancers", "TSS", "TADs", "subTADs", "Loop Domains",
                 "TAD Boundaries", "subTAD Boundaries", "Subcompartments"]
        df_columns = ["Tasks", "GM12878_Rao", "H1hESC_Dekker",
                      "HFFhTERT_Dekker", "GM12878_low", "GM12878_low2"]
        xlabel = "Prediction Target"
        ylabel = "mAP"
        colors = ["C0", "C1", "C4", "C3", "C5"]
        markers = ['o', 'D', 'v', 's', '*']
        labels = ["GM12878 (Rao 2014, 3B)", "H1hESC (Dekker 4DN, 2.5B)", "HFFhTERT (Dekker 4DN, 354M)",
                  "GM12878 (Aiden 4DN, 300M)", "GM12878 (Aiden 4DN, 216M)"]

        "load lists"
        gm_values_all_tasks = np.load(self.path + "gm_reduced_all_tasks.npy")
        h1_values_all_tasks = np.load(self.path + "h1_values_all_tasks.npy")
        hff_values_all_tasks = np.load(self.path + "hff_values_all_tasks.npy")
        gmlow_values_all_tasks = np.load(self.path + "gmlow_metrics_all_tasks.npy")
        gmlow2_values_all_tasks = np.load(self.path + "gmlow2_metrics_all_tasks.npy")

        "setup lists"
        df_lists = [tasks, gm_values_all_tasks, h1_values_all_tasks, gmlow_values_all_tasks, gmlow2_values_all_tasks,
                    hff_values_all_tasks]

        "plot"
        self.plot_main(None, None, df_columns, df_lists, xlabel, ylabel, colors, markers, labels, form_df=True,
                       adjust=True, save=False)

    def plot_map_resolutions(self):
        """
        plot_mAP_resolutions() -> No return object
        Plots mAP across different resolutions
        Args:
            NA
        """

        tasks = ["Gene Expression", "Enhancers", "TADs", "subTADs", "Subcompartments"]
        df_columns = ["Tasks", "lstm_2kbp", "lstm_10kbp",
                      "lstm_100kbp", "lstm_500kbp"]
        xlabel = "Prediction Target"
        ylabel = "mAP"
        colors = ["C1", "C0", "C2", "C3"]
        markers = ['D', 'o', 'v', 's']
        labels = ["2Kbp", "10Kbp", "100Kbp", "500Kbp"]

        "load lists"
        lstm_2kbp = np.load(self.path + "lstm_gm_2kbp.npy")
        lstm_10kbp = np.load(self.path + "lstm_gm_10kbp.npy")
        lstm_100kbp = np.load(self.path + "lstm_gm_100kbp.npy")
        lstm_500kbp = np.load(self.path + "lstm_gm_500kbp.npy")

        "setup lists"
        df_lists = [tasks, lstm_2kbp, lstm_10kbp, lstm_100kbp, lstm_500kbp]

        "plot"
        self.plot_main(df_columns, df_lists, xlabel, ylabel, colors, markers, labels, adjust=True, save=False)

    def plot_auroc_celltypes(self):
        """
        plot_auroc_celltypes() -> No return object
        Plots AuROC for different celltypes
        Args:
            NA
        """

        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains", "Loop Domains", "Subcompartments"]
        df_columns = ["Tasks", "GM12878_Rao", "H1hESC_Dekker", "WTC11_Dekker", "HFFhTERT_Dekker", "GM12878_low",
                      "GM12878_low2"]
        xlabel = "Prediction Target"
        ylabel = "AuROC"
        colors = ["C1", "C0", "C2", "C3"]
        markers = ['o', 'D', '^', 'v', 's', '*']
        labels = ["GM12878 (Rao 2014, 3B)", "H1hESC (Dekker 4DN, 2.5B)", "WTC11 (Dekker 4DN, 1.3B)",
                  "HFFhTERT (Dekker 4DN, 354M)", "GM12878 (Aiden 4DN, 300M)", "GM12878 (Aiden 4DN, 216M)"]

        "load lists"
        gm_auroc_all_tasks = np.load(self.path + "gm_auroc_all_tasks.npy")
        h1_auroc_all_tasks = np.load(self.path + "h1_auroc_all_tasks.npy")
        wtc_auroc_all_tasks = np.load(self.path + "wtc_auroc_all_tasks.npy")
        hff_auroc_all_tasks = np.load(self.path + "hff_auroc_all_tasks.npy")
        gmlow_auroc_all_tasks = np.load(self.path + "gmlow_auroc_all_tasks.npy")
        gmlow2_auroc_all_tasks = np.load(self.path + "gmlow2_auroc_all_tasks.npy")

        "setup lists"
        df_lists = [tasks, gm_auroc_all_tasks, h1_auroc_all_tasks, wtc_auroc_all_tasks, hff_auroc_all_tasks,
                    gmlow_auroc_all_tasks, gmlow2_auroc_all_tasks]

        "plot"
        self.plot_main(df_columns, df_lists, xlabel, ylabel, colors, markers, labels, adjust=False, save=False)

    def plot_auroc(self):
        """
        plot_auroc_celltypes() -> No return object
        Plots AuROC for different celltypes
        Args:
            NA
        """

        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains", "Loop Domains", "Subcompartments"]
        df_columns = ["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA",
                      "SBCID"]
        xlabel = "Prediction Target"
        ylabel = "AuROC"
        colors = ["C3", "C0", "C1", "C2", "C4", "C5"]
        markers = ['o', '*', 'X', '^', 'D', 's']
        labels = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]

        "load lists"
        lstm_auroc_all_tasks = np.load(self.path + "gm_auroc_all_tasks.npy")
        sniper_intra_auroc_all_tasks = np.load(self.path + "sniper_intra_auroc_all_tasks.npy")
        sniper_inter_auroc_all_tasks = np.load(self.path + "sniper_inter_auroc_all_tasks.npy")
        graph_auroc_all_tasks = np.load(self.path + "graph_auroc_all_tasks.npy")
        pca_auroc_all_tasks = np.load(self.path + "pca_auroc_all_tasks.npy")
        sbcid_auroc_all_tasks = np.load(self.path + "sbcid_auroc_all_tasks.npy")

        "setup lists"
        df_lists = [tasks, lstm_auroc_all_tasks, sniper_intra_auroc_all_tasks, sniper_inter_auroc_all_tasks,
                    graph_auroc_all_tasks,
                    pca_auroc_all_tasks, sbcid_auroc_all_tasks]

        "plot"
        self.plot_main(df_columns, df_lists, xlabel, ylabel, colors, markers, labels, adjust=False, save=False)

    def plot_two_axes(self, ax, fig, x_list, y_list, xlabel, ylabel, colors, markers, labels, legend=False,
                      save=False, common=False):

        """
        plot_two_axes(ax, fig, x_list, y_list, xlabel, ylabel, colors, markers, labels, legend, save, common) -> ax, fig, plt
        main function for plotting two axes
        Args:
            ax (AxesSubplot): axes object
            fig (Figure): figure object
            x_list (List): List of x axis values
            y_list (List): List of y axis values
            xlabel (string): xlabel
            ylabel (string): ylabel
            colors (list): List of color values
            markers (list): List of markers
            labels (list): List of labels
            legend (bool): if true makes legend
            save (bool): if true saves figure
            common (bool): if true shares legend
        """

        for i, l in enumerate(labels):
            ax.plot(x_list, y_list[i], marker=markers[i], markersize=16, color=colors[i], linewidth=3,
                    label=l)

        tick_list = [4, 16, 40, 80, 130]
        ax.tick_params(axis="x", labelrotation=90, labelsize=20)
        if not legend:
            ax.tick_params(axis="y", labelsize=20)
        ax.set_xticks(tick_list)
        ax.set_xticklabels(tick_list)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)

        if not common:
            ax.legend(fontsize=18)

        if legend:
            if common:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, fontsize=18, bbox_to_anchor=(0.53, 0.6))

            plt.show()

            if save:
                plt.savefig("/home/kevindsouza/Downloads/y.png")

        return ax, fig

    def plot_hidden(self):
        """
        plot_hidden() -> No return object
        Plots LSTM ablations across number of hidden nodes
        Args:
            NA
        """

        hidden_list = [4, 8, 16, 32, 64, 128]

        lstm_ablation_lists = pd.read_csv(self.path + "lstm_ablation_lists.npy", sep="\t")

        xlabel = "Representation Size"
        colors = ["C0", "C1", "C2", "C4", "C5"]
        markers = ['o', 'D', '^', 's', 'p']
        labels = ['Hi-C-LSTM', 'Hi-C-LSTM, No.Layers: 2', 'Hi-C-LSTM, w/o Layer Norm', 'Hi-C-LSTM, w Dropout',
                  'Hi-C-LSTM, Bidirectional Lstm']

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

        y_list = [lstm_ablation_lists[0], lstm_ablation_lists[1], lstm_ablation_lists[2], lstm_ablation_lists[3],
                  lstm_ablation_lists[4]]
        ylabel = "Avg mAP Across Tasks"
        ax, fig = self.plot_two_axes(ax1, fig, hidden_list, y_list, xlabel, ylabel, colors, markers, labels,
                                     legend=False, save=False, common=True)

        y_list = [lstm_ablation_lists[5], lstm_ablation_lists[6], lstm_ablation_lists[7], lstm_ablation_lists[8],
                  lstm_ablation_lists[9]]
        ylabel = "Avg Hi-C R-squared"
        _, _ = self.plot_two_axes(ax2, fig, hidden_list, y_list, xlabel, ylabel, colors, markers, labels,
                                  legend=True, save=False, common=True)

    def plot_xgb(self):
        """
        plot_xgb() -> No return object
        Plots XGB ablations across number of estimators and depth.
        Args:
            NA
        """

        depth_list = [2, 4, 6, 8, 12, 20]
        estimators_list = [2000, 4000, 5000, 6000, 8000, 10000]

        xgb_lists = pd.read_csv(self.path + "estimators_depth.npy", sep="\t")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        ylabel = "Avg mAP Across Tasks"
        colors = ["C0", "C1", "C2", "C4", "C5"]
        markers = ['o', 's', '^', 'D', 'p']

        y_list = [xgb_lists[0], xgb_lists[1], xgb_lists[2], xgb_lists[3], xgb_lists[4]]
        labels = ['Max Estimators: 2000', 'Max Estimators: 4000', 'Max Estimators: 5000', 'Max Estimators: 6000',
                  'Max Estimators: 10000']
        xlabel = "Max Depth"
        ax, fig = self.plot_two_axes(ax1, fig, depth_list, y_list, xlabel, ylabel, colors, markers, labels,
                                     legend=False, save=False, common=False)

        y_list = [xgb_lists[5], xgb_lists[6], xgb_lists[7], xgb_lists[8], xgb_lists[9]]
        labels = ['Max Depth: 2', 'Max Depth: 4', 'Max Depth: 6', 'Max Depth: 12',
                  'Max Depth: 20']
        xlabel = "Max Estimators"
        _, _ = self.plot_two_axes(ax2, fig, estimators_list, y_list, xlabel, ylabel, colors, markers, labels,
                                  legend=True, save=False, common=False)

    def plot_violin(self):
        """
        plot_violin() -> No return object
        Plots violin plots for segway and TFs
        Args:
            NA
        """

        # ig_log_df = pd.DataFrame(np.load(self.path + "ig_log_df_all.npy", allow_pickle=True))
        ig_log_df = pd.DataFrame(np.load(self.path + "ig_tf_df_plus_ctcf.npy", allow_pickle=True))
        ig_log_df = ig_log_df.rename(columns={0: "ig_val", 1: "label"})
        ig_log_df["ig_val"] = ig_log_df["ig_val"].astype(float)

        plt.figure(figsize=(16, 7))
        sns.set(font_scale=1.8)
        sns.set_style(style='white')
        plt.xticks(rotation=90, fontsize=20)
        plt.ylim(-1, 1)
        ax = sns.violinplot(x="label", y="ig_val", data=ig_log_df)
        ax.set(xlabel='', ylabel='IG Importance')
        plt.show()

    def plot_r2_celltypes(self):
        """
        plot_r2_celltypes() -> No return object
        Plots R2 for Hi-C_LSTM across celltypes
        Args:
            NA
        """

        pos = [10, 20, 30, 40, 50]

        r1_hiclstm_gm = np.load(self.path + "r1_hiclstm_full.npy")
        r1_hiclstm_h1 = np.load(self.path + "r1_hiclstm_h1.npy")
        r1_hiclstm_hff = np.load(self.path + "r1_hiclstm_hff.npy")
        r1_hiclstm_wtc = np.load(self.path + "r1_hiclstm_wtc.npy")
        r1_hiclstm_gmlow = np.load(self.path + "r1_hiclstm_gmlow.npy")
        r1_hiclstm_gmlow2 = np.load(self.path + "r1_hiclstm_gmlow2.npy")

        r2_hiclstm_gm = np.load(self.path + "r2_hiclstm_lstm.npy")
        r2_hiclstm_h1 = np.load(self.path + "r2_hiclstm_h1.npy")
        r2_hiclstm_wtc = np.load(self.path + "r2_hiclstm_wtc.npy")
        r2_hiclstm_gmlow = np.load(self.path + "r2_hiclstm_gmlow.npy")
        r2_hiclstm_hff = np.load(self.path + "r2_hiclstm_hff.npy")
        r2_hiclstm_gmlow2 = np.load(self.path + "r2_hiclstm_gmlow2.npy")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))
        labels = ['GM12878 (Rao 2014, 3B)', 'H1hESC (Dekker 4DN, 2.5B)', 'WTC11 (Dekker 4DN, 1.3B)',
                  'HFFhTERT (Dekker 4DN, 354M)', 'GM12878 (Aiden 4DN, 300M)', 'GM12878 (Aiden 4DN, 216M)']
        xlabel = "Distance between positions in Mbp"
        colors = ["C0", "C1", "C2", "C4", "C3", "C5"]
        markers = ['o', 'D', '^', 'v', 's', '*']

        y_list = [r1_hiclstm_gm, r1_hiclstm_h1, r1_hiclstm_wtc, r1_hiclstm_hff, r1_hiclstm_gmlow, r1_hiclstm_gmlow2]
        ylabel = "R-squared for Replicate-1"
        ax, fig = self.plot_two_axes(ax1, fig, pos, y_list, xlabel, ylabel, colors, markers, labels,
                                     legend=False, save=False, common=True)

        y_list = [r2_hiclstm_gm, r2_hiclstm_h1, r2_hiclstm_wtc, r2_hiclstm_hff, r2_hiclstm_gmlow, r2_hiclstm_gmlow2]
        xlabel = "R-squared for Replicate-2"
        _, _ = self.plot_two_axes(ax2, fig, pos, y_list, xlabel, ylabel, colors, markers, labels,
                                  legend=True, save=False, common=True)

    def plot_r2(self, cell):
        """
        plot_r2() -> No return object
        Plots R2 for various celltypes. Compare methods.
        Args:
            NA
        """
        if cell == "GM12878":
            r1_hiclstm_full = np.load(self.path + "r1_hiclstm_full.npy")
            r1_hiclstm_lstm = np.load(self.path + "r1_hiclstm_lstm.npy")
            r1_hiclstm_cnn = np.load(self.path + "r1_hiclstm_cnn.npy")
            r1_sci_lstm = np.load(self.path + "r1_sci_lstm.npy")
            r1_sniper_lstm = np.load(self.path + "r1_sniper_lstm.npy")
            r1_sci_cnn = np.load(self.path + "r1_sci_cnn.npy")
            r1_sniper_cnn = np.load(self.path + "r1_sniper_cnn.npy")
            r1_hiclstm_fc = np.load(self.path + "r1_hiclstm_fc.npy")
            r1_sci_fc = np.load(self.path + "r1_sci_fc.npy")
            r1_sniper_fc = np.load(self.path + "r1_sniper_fc.npy")

            r2_hiclstm_lstm = np.load(self.path + "r2_hiclstm_lstm.npy")
            r2_hiclstm_cnn = np.load(self.path + "r2_hiclstm_cnn.npy")
            r2_sci_lstm = np.load(self.path + "r2_sci_lstm.npy")
            r2_sniper_lstm = np.load(self.path + "r2_sniper_lstm.npy")
            r2_sci_cnn = np.load(self.path + "r2_sci_cnn.npy")
            r2_sniper_cnn = np.load(self.path + "r2_sniper_cnn.npy")
            r2_hiclstm_fc = np.load(self.path + "r2_hiclstm_fc.npy")
            r2_sci_fc = np.load(self.path + "r2_sci_fc.npy")
            r2_sniper_fc = np.load(self.path + "r2_sniper_fc.npy")
        elif cell == "H1hESC":
            r1_hiclstm_full = np.load(self.path + "r1_hiclstm_h1.npy")
            r1_hiclstm_lstm = np.load(self.path + "r1_hiclstm_lstm_h1.npy")
            r1_hiclstm_cnn = np.load(self.path + "r1_hiclstm_cnn_h1.npy")
            r1_sci_lstm = np.load(self.path + "r1_sci_lstm_h1.npy")
            r1_sniper_lstm = np.load(self.path + "r1_sniper_lstm_h1.npy")
            r1_sci_cnn = np.load(self.path + "r1_sci_cnn_h1.npy")
            r1_sniper_cnn = np.load(self.path + "r1_sniper_cnn_h1.npy")
            r1_hiclstm_fc = np.load(self.path + "r1_hiclstm_fc_h1.npy")
            r1_sci_fc = np.load(self.path + "r1_sci_fc_h1.npy")
            r1_sniper_fc = np.load(self.path + "r1_sniper_fc_h1.npy")

            r2_hiclstm_lstm = np.load(self.path + "r2_hiclstm_h1.npy")
            r2_hiclstm_cnn = np.load(self.path + "r2_hiclstm_cnn_h1.npy")
            r2_sci_lstm = np.load(self.path + "r2_sci_lstm_h1.npy")
            r2_sniper_lstm = np.load(self.path + "r2_sniper_lstm_h1.npy")
            r2_sci_cnn = np.load(self.path + "r2_sci_cnn_h1.npy")
            r2_sniper_cnn = np.load(self.path + "r2_sniper_cnn_h1.npy")
            r2_hiclstm_fc = np.load(self.path + "r2_hiclstm_fc_h1.npy")
            r2_sci_fc = np.load(self.path + "r2_sci_fc_h1.npy")
            r2_sniper_fc = np.load(self.path + "r2_sniper_fc_h1.npy")
        elif cell == "WTC11":
            r1_hiclstm_full = np.load(self.path + "r1_hiclstm_wtc.npy")
            r1_hiclstm_lstm = np.load(self.path + "r1_hiclstm_lstm_wtc.npy")
            r1_hiclstm_cnn = np.load(self.path + "r1_hiclstm_cnn_wtc.npy")
            r1_sci_lstm = np.load(self.path + "r1_sci_lstm_wtc.npy")
            r1_sniper_lstm = np.load(self.path + "r1_sniper_lstm_wtc.npy")
            r1_sci_cnn = np.load(self.path + "r1_sci_cnn_wtc.npy")
            r1_sniper_cnn = np.load(self.path + "r1_sniper_cnn_wtc.npy")
            r1_hiclstm_fc = np.load(self.path + "r1_hiclstm_fc_wtc.npy")
            r1_sci_fc = np.load(self.path + "r1_sci_fc_wtc.npy")
            r1_sniper_fc = np.load(self.path + "r1_sniper_fc_wtc.npy")

            r2_hiclstm_lstm = np.load(self.path + "r2_hiclstm_wtc.npy")
            r2_hiclstm_cnn = np.load(self.path + "r2_hiclstm_cnn_wtc.npy")
            r2_sci_lstm = np.load(self.path + "r2_sci_lstm_wtc.npy")
            r2_sniper_lstm = np.load(self.path + "r2_sniper_lstm_wtc.npy")
            r2_sci_cnn = np.load(self.path + "r2_sci_cnn_wtc.npy")
            r2_sniper_cnn = np.load(self.path + "r2_sniper_cnn_wtc.npy")
            r2_hiclstm_fc = np.load(self.path + "r2_hiclstm_fc_wtc.npy")
            r2_sci_fc = np.load(self.path + "r2_sci_fc_wtc.npy")
            r2_sniper_fc = np.load(self.path + "r2_sniper_fc_wtc.npy")
        elif cell == "HFFhTERT":
            r1_hiclstm_full = np.load(self.path + "r1_hiclstm_hff.npy")
            r1_hiclstm_lstm = np.load(self.path + "r1_hiclstm_lstm_hff.npy")
            r1_hiclstm_cnn = np.load(self.path + "r1_hiclstm_cnn_hff.npy")
            r1_sci_lstm = np.load(self.path + "r1_sci_lstm_hff.npy")
            r1_sniper_lstm = np.load(self.path + "r1_sniper_lstm_hff.npy")
            r1_sci_cnn = np.load(self.path + "r1_sci_cnn_hff.npy")
            r1_sniper_cnn = np.load(self.path + "r1_sniper_cnn_hff.npy")
            r1_hiclstm_fc = np.load(self.path + "r1_hiclstm_fc_hff.npy")
            r1_sci_fc = np.load(self.path + "r1_sci_fc_hff.npy")
            r1_sniper_fc = np.load(self.path + "r1_sniper_fc_hff.npy")

            r2_hiclstm_lstm = np.load(self.path + "r2_hiclstm_hff.npy")
            r2_hiclstm_cnn = np.load(self.path + "r2_hiclstm_cnn_hff.npy")
            r2_sci_lstm = np.load(self.path + "r2_sci_lstm_hff.npy")
            r2_sniper_lstm = np.load(self.path + "r2_sniper_lstm_hff.npy")
            r2_sci_cnn = np.load(self.path + "r2_sci_cnn_hff.npy")
            r2_sniper_cnn = np.load(self.path + "r2_sniper_cnn_hff.npy")
            r2_hiclstm_fc = np.load(self.path + "r2_hiclstm_fc_hff.npy")
            r2_sci_fc = np.load(self.path + "r2_sci_fc_hff.npy")
            r2_sniper_fc = np.load(self.path + "r2_sniper_fc_hff.npy")

        pos = [10, 20, 30, 40, 50]

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))
        labels = ['Hi-C-LSTM-LSTM', 'Hi-C-LSTM-CNN', 'Hi-C-LSTM-FC', 'SCI-LSTM', 'SCI-CNN', 'SCI-FC',
                  'SNIPER-LSTM', 'SNIPER-CNN', 'SNIPER-FC']
        xlabel = "Distance between positions in Mbp"
        colors = ["C0", "C0", "C0", "C1", "C1", "C1", "C2", "C2", "C2"]
        markers = ['D', '^', 's', 'D', '^', 's', 'D', '^', 's']
        style = ["dashed", "dotted", "dashdot", "dashed", "dotted", "dashdot", "dashed", "dotted", "dashdot"]

        y_list = [r1_hiclstm_lstm, r1_hiclstm_cnn, r1_hiclstm_fc, r1_sci_lstm, r1_sci_cnn, r1_sci_fc, r1_sniper_lstm,
                  r1_sniper_cnn, r1_sniper_fc]
        ylabel = "R-squared for Replicate-1"
        ax, fig = self.plot_two_axes(ax1, fig, pos, y_list, xlabel, ylabel, colors, markers, labels,
                                     legend=False, save=False, common=True)

        y_list = [r2_hiclstm_lstm, r2_hiclstm_cnn, r2_hiclstm_fc, r2_sci_lstm, r2_sci_cnn, r2_sci_fc, r2_sniper_lstm,
                  r2_sniper_cnn, r2_sniper_fc]
        xlabel = "R-squared for Replicate-2"
        _, _ = self.plot_two_axes(ax2, fig, pos, y_list, xlabel, ylabel, colors, markers, labels,
                                  legend=True, save=False, common=True)

    def plot_knockout_tfs(self):
        pos = np.linspace(0, 1, 11)
        predicted_probs = np.load(self.path + "predicted_probs.npy")

        ctcfko_probs = np.load(self.path + "ctcfko_probs.npy")
        ctcfko_probs_nl = np.load(self.path + "ctcfko_probs_nl.npy")
        znfko_probs = np.load(self.path + "znfko_probs.npy")
        foxgko_probs = np.load(self.path + "foxgko_probs.npy")
        soxko_probs = np.load(self.path + "soxko_probs.npy")
        xbpko_probs = np.load(self.path + "xbpko_probs.npy")

        # control - KO
        ctcfko_diff = ctcfko_probs - predicted_probs
        ctcfnl_diff = ctcfko_probs_nl - predicted_probs
        znfko_diff = znfko_probs - predicted_probs
        foxgko_diff = foxgko_probs - predicted_probs
        soxko_diff = soxko_probs - predicted_probs
        xbpko_diff = xbpko_probs - predicted_probs

        df_main = pd.DataFrame(
            columns=["pos", "CTCF KO (Loop)", "CTCF KO (Non-loop)", "ZNF143 KO", "FOXG1 KO", "SOX2 KO", "XBP1 KO"])
        df_main["pos"] = pos
        # df_main["No KO"] = predicted_probs
        df_main["CTCF KO (Loop)"] = ctcfko_diff
        df_main["CTCF KO (Non-loop)"] = ctcfnl_diff
        df_main["ZNF143 KO"] = znfko_diff
        df_main["FOXG1 KO"] = foxgko_diff
        df_main["SOX2 KO"] = soxko_diff
        df_main["XBP1 KO"] = xbpko_diff

        palette = {"CTCF KO": "C0", "Convergent CTCF": "C5", "Divergent CTCF": "C1", "RAD21 KO": "C2",
                   "SMC3 KO": "C4"}
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Distance between positions in Mbp", fontsize=20)
        plt.ylabel("Average Difference in Contact Strength \n (KO - No KO)", fontsize=20)
        # plt.plot('pos', 'No KO', data=df_main, marker='o', markersize=14, color="C3", linewidth=2, label="No KO")
        plt.plot('pos', 'CTCF KO (Loop)', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="CTCF KO (Loop)")
        plt.plot('pos', 'CTCF KO (Non-loop)', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 linestyle='dotted', label="CTCF KO (Non-loop)")
        plt.plot('pos', 'ZNF143 KO', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 linestyle='dashed', label="ZNF143 KO")
        plt.plot('pos', 'FOXG1 KO', data=df_main, marker='s', markersize=16, color="C2", linewidth=3,
                 linestyle='dashdot', label="FOXG1 KO")
        plt.plot('pos', 'SOX2 KO', data=df_main, marker='^', markersize=16, color="C4", linewidth=3, label="SOX2 KO")
        plt.plot('pos', 'XBP1 KO', data=df_main, marker='x', markersize=16, color="C6", linewidth=3, label="XBP1 KO")
        plt.legend(fontsize=18)
        plt.show()

        pass

    def plot_knockout_results(self):
        pos = np.linspace(0, 1, 11)
        predicted_probs = np.load(self.path + "predicted_probs.npy")

        ctcfko_probs = np.load(self.path + "ctcfko_probs.npy")
        convctcf_probs = np.load(self.path + "convctcf_probs.npy")
        divctcf_probs = np.load(self.path + "divctcf_probs.npy")
        radko_probs = np.load(self.path + "radko_probs.npy")

        ctcfko_probs_nl = np.load(self.path + "ctcfko_probs_nl.npy")
        znfko_probs = np.load(self.path + "znfko_probs.npy")
        foxgko_probs = np.load(self.path + "foxgko_probs.npy")
        soxko_probs = np.load(self.path + "soxko_probs.npy")
        xbpko_probs = np.load(self.path + "xbpko_probs.npy")

        # control - KO
        ctcfko_diff = ctcfko_probs - predicted_probs
        convctcf_diff = convctcf_probs - predicted_probs
        divctcf_diff = divctcf_probs - predicted_probs

        ctcfnl_diff = ctcfko_probs_nl - predicted_probs
        znfko_diff = znfko_probs - predicted_probs
        foxgko_diff = foxgko_probs - predicted_probs
        soxko_diff = soxko_probs - predicted_probs
        xbpko_diff = xbpko_probs - predicted_probs

        df_main = pd.DataFrame(columns=["pos", "CTCF_Cohesin_KO_Loop", "Convergent_CTCF", "Divergent_CTCF",
                                        "CTCF_Cohesin_KO_nl", "ZNF143_KO", "FOXG1_KO", "SOX2_KO", "XBP1_KO"])
        df_main["pos"] = pos
        df_main["CTCF_Cohesin_KO_Loop"] = ctcfko_diff
        df_main["Convergent_CTCF"] = convctcf_diff
        df_main["Divergent_CTCF"] = divctcf_diff
        df_main["CTCF_Cohesin_KO_nl"] = ctcfnl_diff
        df_main["ZNF143_KO"] = znfko_diff
        df_main["FOXG1_KO"] = foxgko_diff
        df_main["SOX2_KO"] = soxko_diff
        df_main["XBP1_KO"] = xbpko_diff

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Distance between positions in Mbp", fontsize=20)
        plt.ylabel("Average Difference in Contact Strength \n (KO - No KO)", fontsize=20)
        # plt.plot('pos', 'No KO', data=df_main, marker='o', markersize=14, color="C3", linewidth=2, label="No KO")
        plt.plot('pos', 'CTCF_Cohesin_KO_Loop', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="CTCF+Cohesin KO (Loop)")
        plt.plot('pos', 'CTCF_Cohesin_KO_nl', data=df_main, marker='s', markersize=16, color="C2", linewidth=3,
                 linestyle='dotted', label="CTCF+Cohesin KO (Non-loop)")
        plt.plot('pos', 'Convergent_CTCF', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 linestyle='dotted', label="Div->Conv CTCF")
        plt.plot('pos', 'Divergent_CTCF', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 linestyle='dashed', label="Conv->Div CTCF")
        plt.plot('pos', 'ZNF143_KO', data=df_main, marker='^', markersize=16, color="C3", linewidth=3,
                 linestyle='dashed', label="ZNF143 KO")
        plt.plot('pos', 'FOXG1_KO', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 linestyle='dashdot', label="FOXG1 KO")
        plt.plot('pos', 'SOX2_KO', data=df_main, marker='x', markersize=16, color="C6", linewidth=3, label="SOX2 KO")
        plt.plot('pos', 'XBP1_KO', data=df_main, marker='+', markersize=16, color="C7", linewidth=3, label="XBP1 KO")

        plt.legend(fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.savefig("/home/kevindsouza/Downloads/ko_average.png")
        # plt.show()

        pass

    def pr_curves(self):
        path = "/data2/hic_lstm/downstream/"
        precision_file = "precision.npy"
        recall_file = "recall.npy"

        gene_p = np.load(path + "RNA-seq/" + precision_file)
        gene_r = np.load(path + "RNA-seq/" + recall_file)
        rep_p = np.load(path + "replication_timing/" + precision_file)
        rep_r = np.load(path + "replication_timing/" + recall_file)
        pe_p = np.load(path + "PE-interactions/" + precision_file)
        pe_r = np.load(path + "PE-interactions/" + recall_file)
        fire_p = np.load(path + "FIREs/" + precision_file)
        fire_r = np.load(path + "FIREs/" + recall_file)
        tss_p = np.load(path + "tss/" + precision_file)
        tss_r = np.load(path + "tss/" + recall_file)
        en_p = np.load(path + "enhancers/" + precision_file)
        en_r = np.load(path + "enhancers/" + recall_file)
        domain_p = np.load(path + "domains/" + precision_file)
        domain_r = np.load(path + "domains/" + recall_file)
        subtad_p = np.load(path + "domains/" + "precision_sub.npy")
        subtad_r = np.load(path + "domains/" + "recall_sub.npy")
        loop_p = np.load(path + "loops/" + precision_file)
        loop_r = np.load(path + "loops/" + recall_file)
        subc_p = np.load(path + "subcompartments/" + precision_file)
        subc_r = np.load(path + "subcompartments/" + recall_file)

        plt.figure(figsize=(8, 6))
        plt.step(gene_r, gene_p, color='b', alpha=0.5, where='post', linewidth=3, label="Gene Expression")
        plt.step(rep_r, rep_p, color='g', alpha=0.5, where='post', linewidth=3, label="Replication Timing")
        plt.step(en_r, en_p, color='r', alpha=0.5, where='post', linewidth=3, label="Enhancers")
        plt.step(tss_r, tss_p, color='c', alpha=0.5, where='post', linewidth=3, label="TSS")
        plt.step(pe_r, pe_p, color='m', alpha=0.5, where='post', linewidth=3, label="PE-Interactions")
        plt.step(fire_r, fire_p, color='y', alpha=0.5, where='post', linewidth=3, label="FIREs")
        plt.step(domain_r, domain_p, color='k', alpha=0.5, where='post', linewidth=3, label="TADs")
        plt.step(subtad_r, subtad_p, color='C8', alpha=0.5, where='post', linewidth=3, label="subTADs")
        plt.step(loop_r, loop_p, color='indigo', alpha=0.5, where='post', linewidth=3, label="Loop Domains")
        plt.step(subc_r, subc_p, color='brown', alpha=0.5, where='post', linewidth=3, label="Subcompartments")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1])
        plt.legend(fontsize=14)
        plt.show()

        pass

    def plot_symmetry(self):

        mode = "hist"
        if mode == "hist":
            sym_hist = np.load(self.path + "lstm/" + "symmetry_hist.npy")
            count, bins, ignored = plt.hist(sym_hist, 100, density=True, color='blue', edgecolor='black')
            plt.xlabel("Difference in Contact Strength (Predicted - Original)", fontsize=14)
            plt.ylabel("Normalized Density", fontsize=14)
            plt.yticks([])
            plt.show()
        elif mode == "diff":
            pass

        pass

    def plot_feature_signal(self):

        mode = "tad"

        if mode == "tad":
            pos = np.arange(-110, 120, 10)
            plt.figure(figsize=(6, 2))
            feature_signal = np.load(self.path + "tad_ig_signal.npy")
            plt.bar(pos, feature_signal, width=4, color='g')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Distance in Kbp", fontsize=14)
            plt.ylabel("IG Importance", fontsize=14)
            plt.show()

            pass
        elif mode == "chr21":
            pos = np.arange(28, 29.2, 0.025)
            feature_signal_chr21 = np.load(self.path + "lstm/" + "feature_signal_chr21.npy")
            plt.figure(figsize=(6, 2))
            plt.plot(pos, feature_signal_chr21)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Positions in Mbp", fontsize=14)
            plt.ylabel("IG Importance", fontsize=14)
            plt.show()
            pass

        pass

    def plot_pred_range(self):
        chr = 21

        diff_list = [0.12, 0.114, 0.105, 0.0714, 0.048, 0.002, -0.021, -0.043, -0.067, -0.082, -0.096]
        og_list = np.arange(0, 1.1, 0.1)

        plt.plot(og_list, diff_list, linewidth=3, marker='o', markersize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Original Hi-C Contact Strength", fontsize=14)
        plt.ylabel("Difference in Contact Strength \n (Predicted - Original)", fontsize=14)
        plt.show()
        pass


if __name__ == "__main__":
    cfg = Config()
    plot_ob = PlotFns(cfg)

    # plot_ob.plot_combined(cell="GM12878", metric="accuracy", ylabel="Accuracy")
    # plot_ob.plot_map_celltypes()
    plot_ob.plot_map_resolutions()
    # plot_ob.plot_auroc_celltypes()
    # plot_ob.plot_auroc()
    # plot_ob.plot_hidden()
    # plot_ob.plot_xgb()
    # plot_ob.plot_violin()
    # plot_ob.plot_r2_celltypes()
    # plot_ob.plot_r2(cell="GM12878")
    # plot_ob.plot_symmetry()
    # plot_ob.plot_knockout_results()
    # plot_ob.plot_knockout_tfs()
    # plot_ob.pr_curves()
    # plot_ob.plot_feature_signal()
    # plot_ob.plot_pred_range()

    print("done")
