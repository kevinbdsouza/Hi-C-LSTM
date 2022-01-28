import logging
import numpy as np
import matplotlib as mpl

# mpl.use('WebAgg')
mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt
import operator
import pandas as pd
import seaborn as sns
import training.config as config
from analyses.classification.domains import Domains
from training.data_utils import get_cumpos

logger = logging.getLogger(__name__)


class PlotFns:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.output_directory
        self.cell_type = ["E116", "GM12878"]

    def get_dict(self, path):
        lstm_rna = np.load(path + "lstm/map_dict_rnaseq.npy").item()
        sniper_rna = np.load(path + "sniper/map_dict_rnaseq.npy").item()
        graph_rna = np.load(path + "graph/map_dict_rnaseq.npy").item()
        pca_rna = np.load(path + "pca/map_dict_rnaseq.npy").item()
        sbcid_rna = np.load(path + "sbcid/map_dict_rnaseq.npy").item()

        lstm_pe = np.load(path + "lstm/map_dict_pe.npy").item()
        sniper_pe = np.load(path + "sniper/map_dict_pe.npy").item()
        graph_pe = np.load(path + "graph/map_dict_pe.npy").item()
        pca_pe = np.load(path + "pca/map_dict_pe.npy").item()
        sbcid_pe = np.load(path + "sbcid/map_dict_pe.npy").item()

        lstm_fire = np.load(path + "lstm/map_dict_fire.npy").item()
        sniper_fire = np.load(path + "sniper/map_dict_fire.npy").item()
        graph_fire = np.load(path + "graph/map_dict_fire.npy").item()
        pca_fire = np.load(path + "pca/map_dict_fire.npy").item()
        sbcid_fire = np.load(path + "sbcid/map_dict_fire.npy").item()

        lstm_rep = np.load(path + "lstm/map_dict_rep.npy").item()
        sniper_rep = np.load(path + "sniper/map_dict_rep.npy").item()
        graph_rep = np.load(path + "graph/map_dict_rep.npy").item()
        pca_rep = np.load(path + "pca/map_dict_rep.npy").item()
        sbcid_rep = np.load(path + "sbcid/map_dict_rep.npy").item()

        lstm_promoter = np.load(path + "lstm/map_dict_promoter.npy").item()
        sniper_promoter = np.load(path + "sniper/map_dict_promoter.npy").item()
        graph_promoter = np.load(path + "graph/map_dict_promoter.npy").item()
        pca_promoter = np.load(path + "pca/map_dict_promoter.npy").item()
        sbcid_promoter = np.load(path + "sbcid/map_dict_promoter.npy").item()

        lstm_enhancer = np.load(path + "lstm/map_dict_enhancer.npy").item()
        sniper_enhancer = np.load(path + "sniper/map_dict_enhancer.npy").item()
        graph_enhancer = np.load(path + "graph/map_dict_enhancer.npy").item()
        pca_enhancer = np.load(path + "pca/map_dict_enhancer.npy").item()
        sbcid_enhancer = np.load(path + "sbcid/map_dict_enhancer.npy").item()

        lstm_domain = np.load(path + "lstm/map_dict_domain.npy").item()
        sniper_domain = np.load(path + "sniper/map_dict_domain.npy").item()
        graph_domain = np.load(path + "graph/map_dict_domain.npy").item()
        pca_domain = np.load(path + "pca/map_dict_domain.npy").item()
        sbcid_domain = np.load(path + "sbcid/map_dict_domain.npy").item()

        lstm_loop = np.load(path + "lstm/map_dict_loop.npy").item()
        sniper_loop = np.load(path + "sniper/map_dict_loop.npy").item()
        graph_loop = np.load(path + "graph/map_dict_loop.npy").item()
        pca_loop = np.load(path + "pca/map_dict_loop.npy").item()
        sbcid_loop = np.load(path + "sbcid/map_dict_loop.npy").item()

        lstm_tss = np.load(path + "lstm/map_dict_tss.npy").item()
        sniper_tss = np.load(path + "sniper/map_dict_tss.npy").item()
        graph_tss = np.load(path + "graph/map_dict_tss.npy").item()
        pca_tss = np.load(path + "pca/map_dict_tss.npy").item()
        sbcid_tss = np.load(path + "sbcid/map_dict_tss.npy").item()

        lstm_sbc = np.load(path + "lstm/map_dict_sbc.npy").item()
        sniper_sbc = np.load(path + "sniper/map_dict_sbc.npy").item()
        graph_sbc = np.load(path + "graph/map_dict_sbc.npy").item()
        pca_sbc = np.load(path + "pca/map_dict_sbc.npy").item()
        sbcid_sbc = np.load(path + "sbcid/map_dict_sbc.npy").item()

        return sniper_rna, sniper_pe, sniper_fire, sniper_rep, sniper_promoter, sniper_enhancer, \
               sniper_domain, sniper_loop, sniper_tss, sniper_sbc, lstm_rna, lstm_pe, lstm_fire, \
               lstm_rep, lstm_promoter, lstm_enhancer, lstm_domain, lstm_loop, lstm_tss, lstm_sbc, \
               graph_rna, graph_pe, graph_fire, graph_rep, graph_promoter, graph_enhancer, graph_domain, \
               graph_loop, graph_tss, graph_sbc, pca_rna, pca_pe, pca_fire, pca_rep, pca_promoter, pca_enhancer, \
               pca_domain, pca_loop, pca_tss, pca_sbc, sbcid_rna, sbcid_pe, sbcid_fire, sbcid_rep, sbcid_promoter, \
               sbcid_enhancer, sbcid_domain, sbcid_loop, sbcid_tss, sbcid_sbc

    def get_lists(self, dict):
        key_list = []
        value_list = []

        for key, value in sorted(dict.items(), key=operator.itemgetter(1), reverse=True):
            key_list.append(key)
            value_list.append(value)

        return key_list, value_list

    def reorder_lists(self, key_list_lstm, key_list_other, value_list_other):
        key_list_other_sort = sorted(key_list_other, key=lambda i: key_list_lstm.index(i))
        temp = {val: key for key, val in enumerate(key_list_other_sort)}
        res = list(map(temp.get, key_list_lstm))
        value_list_other = [value_list_other[i] for i in res]

        return value_list_other

    def plot_heatmaps(self, data):
        st = int(data["i"].min())
        data["i"] = data["i"] - st
        data["j"] = data["j"] - st
        nr = int(data["j"].max()) + 1
        rows = np.array(data["i"]).astype(int)
        cols = np.array(data["j"]).astype(int)

        hic_mat = np.zeros((nr, nr))
        hic_mat[rows, cols] = np.array(data["v"])
        hic_upper = np.triu(hic_mat)
        hic_mat[cols, rows] = np.array(data["pred"])
        hic_lower = np.tril(hic_mat)
        hic_mat = hic_upper + hic_lower
        hic_mat[np.diag_indices_from(hic_mat)] /= 2
        # self.simple_plot(hic_mat)
        return hic_mat, st

    def simple_plot(self, hic_win):
        plt.imshow(hic_win, cmap='hot', interpolation='nearest')
        plt.yticks([])
        plt.xticks([])
        plt.show()
        '''
        sns.set_theme()
        ax = sns.heatmap(hic_win, cmap="Reds")
        ax.set_yticks([])
        ax.set_xticks([])
        plt.show()
        '''
        pass

    def ctcf_dots(self, hic_mat, st, chr):
        dom_ob = Domains(cfg, self.cfg.cell, chr)
        dom_data = dom_ob.get_domain_data()

        th = 41
        mean_map_og = np.zeros((th, th))
        mean_map_pred = np.zeros((th, th))
        num = 0
        for n in range(len(dom_data)):
            x1 = dom_data.loc[n]["x1"] - st + get_cumpos(self.cfg, chr)
            x2 = dom_data.loc[n]["x2"] - st + get_cumpos(self.cfg, chr)
            y1 = dom_data.loc[n]["y1"] - st + get_cumpos(self.cfg, chr)
            y2 = dom_data.loc[n]["y2"] - st + get_cumpos(self.cfg, chr)

            if (x2 - x1) <= th - 1:
                continue
            else:
                num += 1
                hic_win_og = hic_mat[x1:x1 + th, y2 - th:y2]
                hic_win_pred = hic_mat[x1 - th:x1, y1:y1 + th]
                mean_map_og = mean_map_og + hic_win_og
                mean_map_pred = mean_map_pred + hic_win_pred

        mean_map_og = mean_map_og / num
        mean_map_pred = mean_map_pred / num
        self.simple_plot(mean_map_og)
        self.simple_plot(mean_map_pred)

        pass

    def plot_combined_all(self, cell):
        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "TADs", "subTADs", "Loop Domains", "TAD Boundaries", "subTAD Boundaries", "Subcompartments"]

        methods = ["Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"]
        colors = ['C3', 'C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

        if cell == "GM12878":
            lstm_values_all_tasks = np.load(self.path + "lstm_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_values_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_values_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_values_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_values_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_values_all_tasks.npy")

            lstm_auroc_all_tasks = np.load(self.path + "gm_auroc_all_tasks.npy")
            sniper_intra_auroc_all_tasks = np.load(self.path + "sniper_intra_auroc_all_tasks.npy")
            sniper_inter_auroc_all_tasks = np.load(self.path + "sniper_inter_auroc_all_tasks.npy")
            graph_auroc_all_tasks = np.load(self.path + "graph_auroc_all_tasks.npy")
            pca_auroc_all_tasks = np.load(self.path + "pca_auroc_all_tasks.npy")
            sbcid_auroc_all_tasks = np.load(self.path + "sbcid_auroc_all_tasks.npy")

            lstm_accuracy_all_tasks = np.load(self.path + "gm_accuracy_all_tasks.npy")
            sniper_intra_accuracy_all_tasks = np.load(self.path + "gm_sniper_intra_accuracy_all_tasks.npy")
            sniper_inter_accuracy_all_tasks = np.load(self.path + "gm_sniper_inter_accuracy_all_tasks.npy")
            graph_accuracy_all_tasks = np.load(self.path + "gm_graph_accuracy_all_tasks.npy")
            pca_accuracy_all_tasks = np.load(self.path + "gm_pca_accuracy_all_tasks.npy")
            sbcid_accuracy_all_tasks = np.load(self.path + "gm_sbcid_accuracy_all_tasks.npy")

            lstm_fscore_all_tasks = np.load(self.path + "gm_fscore_all_tasks.npy")
            sniper_intra_fscore_all_tasks = np.load(self.path + "gm_sniper_intra_fscore_all_tasks.npy")
            sniper_inter_fscore_all_tasks = np.load(self.path + "gm_sniper_inter_fscore_all_tasks.npy")
            graph_fscore_all_tasks = np.load(self.path + "gm_graph_fscore_all_tasks.npy")
            pca_fscore_all_tasks = np.load(self.path + "gm_pca_fscore_all_tasks.npy")
            sbcid_fscore_all_tasks = np.load(self.path + "gm_sbcid_fscore_all_tasks.npy")

            df_main = pd.DataFrame(
                columns=["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA",
                         "SBCID"])
            df_main["Tasks"] = tasks
            df_main[
                "Hi-C-LSTM"] = lstm_values_all_tasks + lstm_auroc_all_tasks + lstm_accuracy_all_tasks + lstm_fscore_all_tasks
            df_main[
                "SNIPER-INTRA"] = sniper_intra_values_all_tasks + sniper_intra_auroc_all_tasks + sniper_intra_accuracy_all_tasks + sniper_intra_fscore_all_tasks
            df_main[
                "SNIPER-INTER"] = sniper_inter_values_all_tasks + sniper_inter_auroc_all_tasks + sniper_inter_accuracy_all_tasks + sniper_inter_fscore_all_tasks
            df_main[
                "SCI"] = graph_values_all_tasks + graph_auroc_all_tasks + graph_accuracy_all_tasks + graph_fscore_all_tasks
            df_main["PCA"] = pca_values_all_tasks + pca_auroc_all_tasks + pca_accuracy_all_tasks + pca_fscore_all_tasks
            df_main[
                "SBCID"] = sbcid_values_all_tasks + sbcid_auroc_all_tasks + sbcid_accuracy_all_tasks + sbcid_fscore_all_tasks

        #df_main.to_csv(self.path + "%s_metrics_df.csv" % (cell), sep="\t")
        df_main = pd.read_csv(self.path + "%s_metrics_df.csv" % (cell), sep="\t")
        df_main = df_main.drop(['Unnamed: 0'], axis=1)

        def plot_stackedbar(df_main, tasks, colors):
            # df_main = df_main.set_index("Tasks")
            df_main = df_main.T
            df_main.columns = df_main.iloc[0]
            df_main = df_main.drop(["Tasks"], axis=0)
            fields = df_main.columns.tolist()

            # figure and axis
            fig, ax = plt.subplots(1, figsize=(20, 12))

            # plot bars
            left = len(df_main) * [0]
            for idx, name in enumerate(fields):
                plt.barh(df_main.index, df_main[name], left=left, color=colors[idx])
                left = left + df_main[name]

            # legend
            plt.rcParams.update({'font.size': 22})
            plt.legend(tasks, bbox_to_anchor=([0.02, 1, 0, 0]), ncol=6, frameon=False, fontsize=14)

            # remove spines
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # format x ticks
            xticks = np.arange(0, 48.1, 4)
            xlabels = ['{}'.format(i) for i in np.arange(0, 48.1, 4)]
            plt.xticks(xticks, xlabels, fontsize=20)

            # adjust limits and draw grid lines
            plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
            ax.xaxis.grid(color='gray', linestyle='dashed')
            plt.xlabel("Prediction Target", fontsize=20)
            plt.ylabel("Cumulative Prediction Score", fontsize=20)

            plt.show()

        plot_stackedbar(df_main, tasks, colors)

        print("done")

        pass

    def plot_combined(self, cell):
        #tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
        #         "Non-loop Domains", "Loop Domains", "Subcompartments"]

        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "TADs", "subTADs", "Loop Domains", "TAD Boundaries", "subTAD Boundaries", "Subcompartments"]
        colors = ['C3', 'C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

        if cell == "GM12878":
            lstm_values_all_tasks = np.load(self.path + "gm_accuracy_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "gm_sniper_intra_accuracy_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "gm_sniper_inter_accuracy_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "gm_graph_accuracy_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "gm_pca_accuracy_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "gm_sbcid_accuracy_all_tasks.npy")
        elif cell == "H1hESC":
            lstm_values_all_tasks = np.load(self.path + "h1_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_h1_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_h1_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_h1_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_h1_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_h1_all_tasks.npy")
        elif cell == "HFFhTERT":
            lstm_values_all_tasks = np.load(self.path + "hff_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_hff_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_hff_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_hff_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_hff_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_hff_all_tasks.npy")
        elif cell == "WTC11":
            lstm_values_all_tasks = np.load(self.path + "wtc_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_wtc_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_wtc_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_wtc_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_wtc_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_wtc_all_tasks.npy")

        df_main = pd.DataFrame(columns=["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"])
        df_main["Tasks"] = tasks
        df_main["Hi-C-LSTM"] = lstm_values_all_tasks
        df_main["SNIPER-INTRA"] = sniper_intra_values_all_tasks
        df_main["SNIPER-INTER"] = sniper_inter_values_all_tasks
        df_main["SCI"] = graph_values_all_tasks
        df_main["PCA"] = pca_values_all_tasks
        df_main["SBCID"] = sbcid_values_all_tasks

        df_main = pd.read_csv(self.path + "%s_metrics_df.csv" % (cell), sep="\t")
        df_main = df_main.drop(['Unnamed: 0'], axis=1)

        plt.figure(figsize=(12, 10))
        # plt.tight_layout()
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("mAP ", fontsize=20)
        plt.plot('Tasks', 'Hi-C-LSTM', data=df_main, marker='o', markersize=16, color="C3", linewidth=3,
                 label="Hi-C-LSTM")
        plt.plot('Tasks', 'SNIPER-INTRA', data=df_main, marker='*', markersize=16, color="C0", linewidth=3,
                 linestyle='dashed', label="SNIPER-INTRA")
        plt.plot('Tasks', 'SNIPER-INTER', data=df_main, marker='X', markersize=16, color="C1", linewidth=3,
                 linestyle='dotted', label="SNIPER-INTER")
        plt.plot('Tasks', 'SCI', data=df_main, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dashdot',
                 label="SCI")
        plt.plot('Tasks', 'PCA', data=df_main, marker='D', markersize=16, color="C4", linewidth=3, label="PCA")
        plt.plot('Tasks', 'SBCID', data=df_main, marker='s', markersize=16, color="C5", linewidth=3, linestyle='dashed',
                 label="SBCID")
        plt.legend(fontsize=18)
        # plt.savefig("/home/kevindsouza/Downloads/map.png")
        plt.show()

        pass

    def plot_mAP_celltypes(self):
        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains",
                 "Loop Domains", "Subcompartments"]

        gm_values_all_tasks = np.load(self.path + "lstm_values_all_tasks.npy")
        h1_values_all_tasks = np.load(self.path + "h1_values_all_tasks.npy")
        hff_values_all_tasks = np.load(self.path + "hff_values_all_tasks.npy")
        wtc_values_all_tasks = np.load(self.path + "wtc_values_all_tasks.npy")
        gmlow_values_all_tasks = np.load(self.path + "gmlow_values_all_tasks.npy")
        gmlow2_values_all_tasks = np.load(self.path + "gmlow2_values_all_tasks.npy")

        df_main = pd.DataFrame(columns=["Tasks", "GM12878_Rao", "H1hESC_Dekker", "WTC11_Dekker",
                                        "GM12878_low", "HFFhTERT_Dekker", "GM12878_low2"])
        df_main["Tasks"] = tasks
        df_main["GM12878_Rao"] = gm_values_all_tasks
        df_main["H1hESC_Dekker"] = h1_values_all_tasks
        df_main["WTC11_Dekker"] = wtc_values_all_tasks
        df_main["GM12878_low"] = gmlow_values_all_tasks
        df_main["GM12878_low2"] = gmlow2_values_all_tasks
        df_main["HFFhTERT_Dekker"] = hff_values_all_tasks

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("mAP ", fontsize=20)
        plt.plot('Tasks', 'GM12878_Rao', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="GM12878 (Rao 2014, 3B)")
        plt.plot('Tasks', 'H1hESC_Dekker', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 label="H1hESC (Dekker 4DN, 2.5B)")
        plt.plot('Tasks', 'WTC11_Dekker', data=df_main, marker='^', markersize=16, color="C2", linewidth=3,
                 label="WTC11 (Dekker 4DN, 1.3B)")
        plt.plot('Tasks', 'HFFhTERT_Dekker', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 label="HFFhTERT (Dekker 4DN, 354M)")
        plt.plot('Tasks', 'GM12878_low', data=df_main, marker='s', markersize=16, color="C3", linewidth=3,
                 label="GM12878 (Aiden 4DN, 300M)")
        plt.plot('Tasks', 'GM12878_low2', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 label="GM12878 (Aiden 4DN, 216M)")
        plt.legend(fontsize=18)
        plt.savefig("/home/kevindsouza/Downloads/map_cells.png")
        plt.show()

        pass

    def plot_auroc_celltypes(self):
        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains",
                 "Loop Domains", "Subcompartments"]

        gm_auroc_all_tasks = np.load(self.path + "gm_auroc_all_tasks.npy")
        h1_auroc_all_tasks = np.load(self.path + "h1_auroc_all_tasks.npy")
        hff_auroc_all_tasks = np.load(self.path + "hff_auroc_all_tasks.npy")
        wtc_auroc_all_tasks = np.load(self.path + "wtc_auroc_all_tasks.npy")
        gmlow_auroc_all_tasks = np.load(self.path + "gmlow_auroc_all_tasks.npy")
        gmlow2_auroc_all_tasks = np.load(self.path + "gmlow2_auroc_all_tasks.npy")

        df_main = pd.DataFrame(columns=["Tasks", "GM12878_Rao", "H1hESC_Dekker", "WTC11_Dekker",
                                        "GM12878_low", "HFFhTERT_Dekker", "GM12878_low2"])
        df_main["Tasks"] = tasks
        df_main["GM12878_Rao"] = gm_auroc_all_tasks
        df_main["H1hESC_Dekker"] = h1_auroc_all_tasks
        df_main["WTC11_Dekker"] = wtc_auroc_all_tasks
        df_main["GM12878_low"] = gmlow_auroc_all_tasks
        df_main["HFFhTERT_Dekker"] = hff_auroc_all_tasks
        df_main["GM12878_low2"] = gmlow2_auroc_all_tasks

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("AuROC ", fontsize=20)
        plt.plot('Tasks', 'GM12878_Rao', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="GM12878 (Rao 2014, 3B)")
        plt.plot('Tasks', 'H1hESC_Dekker', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 label="H1hESC (Dekker 4DN, 2.5B)")
        plt.plot('Tasks', 'WTC11_Dekker', data=df_main, marker='^', markersize=16, color="C2", linewidth=3,
                 label="WTC11 (Dekker 4DN, 1.3B)")
        plt.plot('Tasks', 'HFFhTERT_Dekker', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 label="HFFhTERT (Dekker 4DN, 354M)")
        plt.plot('Tasks', 'GM12878_low', data=df_main, marker='s', markersize=16, color="C3", linewidth=3,
                 label="GM12878 (Aiden 4DN, 300M)")
        plt.plot('Tasks', 'GM12878_low2', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 label="GM12878 (Aiden 4DN, 216M)")

        plt.legend(fontsize=18)
        plt.show()
        pass

    def plot_auroc(self):
        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains",
                 "Loop Domains", "Subcompartments"]

        lstm_auroc_all_tasks = np.load(self.path + "gm_auroc_all_tasks.npy")
        sniper_intra_auroc_all_tasks = np.load(self.path + "sniper_intra_auroc_all_tasks.npy")
        sniper_inter_auroc_all_tasks = np.load(self.path + "sniper_inter_auroc_all_tasks.npy")
        graph_auroc_all_tasks = np.load(self.path + "graph_auroc_all_tasks.npy")
        pca_auroc_all_tasks = np.load(self.path + "pca_auroc_all_tasks.npy")
        sbcid_auroc_all_tasks = np.load(self.path + "sbcid_auroc_all_tasks.npy")

        df_main = pd.DataFrame(columns=["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"])
        df_main["Tasks"] = tasks
        df_main["Hi-C-LSTM"] = lstm_auroc_all_tasks
        df_main["SNIPER-INTRA"] = sniper_intra_auroc_all_tasks
        df_main["SNIPER-INTER"] = sniper_inter_auroc_all_tasks
        df_main["SCI"] = graph_auroc_all_tasks
        df_main["PCA"] = pca_auroc_all_tasks
        df_main["SBCID"] = sbcid_auroc_all_tasks

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("AuROC ", fontsize=20)
        plt.plot('Tasks', 'Hi-C-LSTM', data=df_main, marker='o', markersize=16, color="C3", linewidth=3,
                 label="Hi-C-LSTM")
        plt.plot('Tasks', 'SNIPER-INTRA', data=df_main, marker='*', markersize=16, color="C0", linewidth=3,
                 linestyle='dashed', label="SNIPER-INTRA")
        plt.plot('Tasks', 'SNIPER-INTER', data=df_main, marker='X', markersize=16, color="C1", linewidth=3,
                 linestyle='dotted', label="SNIPER-INTER")
        plt.plot('Tasks', 'SCI', data=df_main, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dashdot',
                 label="SCI")
        plt.plot('Tasks', 'PCA', data=df_main, marker='D', markersize=16, color="C4", linewidth=3, label="PCA")
        plt.plot('Tasks', 'SBCID', data=df_main, marker='s', markersize=16, color="C5", linewidth=3, linestyle='dashed',
                 label="SBCID")
        plt.legend(fontsize=18)
        plt.show()
        pass

    def plot_hidden(self, hidden_list):
        map_hidden = np.load(self.path + "lstm/" + "hiclstm_ablation.npy")
        map_2_layer = np.load(self.path + "lstm/" + "hiclstm_2_layer_ablation.npy")
        map_bidir = np.load(self.path + "lstm/" + "hiclstm_bidir_ablation.npy")
        map_dropout = np.load(self.path + "lstm/" + "hiclstm_dropout_ablation.npy")
        map_no_ln = np.load(self.path + "lstm/" + "hiclstm_no_ln_ablation.npy")

        r2_hidden = np.load(self.path + "lstm/" + "hiclstm_ablation_r2.npy")
        r2_bidir = np.load(self.path + "lstm/" + "hiclstm_bidir_ablation_r2.npy")
        r2_2_layer = np.load(self.path + "lstm/" + "hiclstm_2_layer_ablation_r2.npy")
        r2_dropout = np.load(self.path + "lstm/" + "hiclstm_dropout_ablation_r2.npy")
        r2_no_ln = np.load(self.path + "lstm/" + "hiclstm_no_ln_ablation_r2.npy")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

        ax1.plot(hidden_list, map_hidden, marker='o', markersize=16, color="C0", linewidth=3,
                 label='Hi-C-LSTM')
        ax1.plot(hidden_list, map_2_layer, marker='D', markersize=16, color="C1", linewidth=3, linestyle='dashed',
                 label='Hi-C-LSTM, No.Layers: 2')
        ax1.plot(hidden_list, map_no_ln, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dotted',
                 label='Hi-C-LSTM, w/o Layer Norm')
        ax1.plot(hidden_list, map_dropout, marker='s', markersize=16, color="C4", linewidth=3, linestyle='dashdot',
                 label='Hi-C-LSTM, w Dropout')
        ax1.plot(hidden_list, map_bidir, marker='p', markersize=16, color="C5", linewidth=3, label='Hi-C-LSTM, '
                                                                                                   'Bidirectional Lstm')

        ax2.plot(hidden_list, r2_hidden, marker='o', markersize=16, color="C0", linewidth=3,
                 label='Hi-C-LSTM')
        ax2.plot(hidden_list, r2_2_layer, marker='D', markersize=16, color="C1", linewidth=3, linestyle='dashed',
                 label='Hi-C-LSTM, No.Layers: 2')
        ax2.plot(hidden_list, r2_no_ln, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dotted',
                 label='Hi-C-LSTM, w/o Layer Norm')
        ax2.plot(hidden_list, r2_dropout, marker='s', markersize=16, color="C4", linewidth=3, linestyle='dashdot',
                 label='Hi-C-LSTM, w Dropout')
        ax2.plot(hidden_list, r2_bidir, marker='p', markersize=16, color="C5", linewidth=3, label='Hi-C-LSTM, '
                                                                                                  'Bidirectional Lstm')

        ax1.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax2.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax1.tick_params(axis="y", labelsize=20)
        tick_list = [4, 16, 40, 80, 130]
        ax1.set_xticks(tick_list)
        ax1.set_xticklabels(tick_list)
        ax2.set_xticks(tick_list)
        ax2.set_xticklabels(tick_list)
        ax1.set_xlabel('Representation Size', fontsize=20)
        ax2.set_xlabel('Representation Size', fontsize=20)
        ax1.set_ylabel('Avg mAP Across Tasks', fontsize=20)
        ax2.set_ylabel('Avg Hi-C R-squared', fontsize=20)

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='best', fontsize=18, bbox_to_anchor=(0.53, 0.6))

        plt.show()

        pass

    def plot_xgb(self):
        depth_list = [2, 4, 6, 8, 12, 20]
        estimators_list = [2000, 4000, 5000, 6000, 8000, 10000]

        plt.figure(figsize=(10, 6))
        map_depth_2000 = np.load(self.path + "lstm/" + "xgb_map_depth_2000.npy")
        map_depth_4000 = np.load(self.path + "lstm/" + "xgb_map_depth_4000.npy")
        map_depth_5000 = np.load(self.path + "lstm/" + "xgb_map_depth_5000.npy")
        map_depth_6000 = np.load(self.path + "lstm/" + "xgb_map_depth_6000.npy")
        map_depth_10000 = np.load(self.path + "lstm/" + "xgb_map_depth_10000.npy")

        map_est_2 = np.load(self.path + "lstm/" + "xgb_map_est_2.npy")
        map_est_4 = np.load(self.path + "lstm/" + "xgb_map_est_4.npy")
        map_est_6 = np.load(self.path + "lstm/" + "xgb_map_est_6.npy")
        map_est_12 = np.load(self.path + "lstm/" + "xgb_map_est_12.npy")
        map_est_20 = np.load(self.path + "lstm/" + "xgb_map_est_20.npy")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

        ax1.plot(depth_list, map_depth_2000, marker='o', markersize=16, color="C0", linewidth=3,
                 label='Max Estimators: 2000')
        ax1.plot(depth_list, map_depth_4000, marker='s', markersize=16, color="C1", linewidth=3, linestyle='dashed',
                 label='Max Estimators: 4000')
        ax1.plot(depth_list, map_depth_5000, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dotted',
                 label='Max Estimators: 5000')
        ax1.plot(depth_list, map_depth_6000, marker='D', markersize=16, color="C4", linewidth=3, linestyle='dashdot',
                 label='Max Estimators: 6000')
        ax1.plot(depth_list, map_depth_10000, marker='p', markersize=16, color="C5", linewidth=3,
                 label='Max Estimators: 10000')

        ax2.plot(estimators_list, map_est_2, marker='o', markersize=16, color="C0", linewidth=3,
                 label='Max Depth: 2')
        ax2.plot(estimators_list, map_est_4, marker='s', markersize=16, color="C1", linewidth=3, linestyle='dashed',
                 label='Max Depth: 4')
        ax2.plot(estimators_list, map_est_6, marker='^', markersize=16, color="C2", linewidth=3, linestyle='dotted',
                 label='Max Depth: 6')
        ax2.plot(estimators_list, map_est_12, marker='D', markersize=16, color="C4", linewidth=3, linestyle='dashdot',
                 label='Max Depth: 12')
        ax2.plot(estimators_list, map_est_20, marker='p', markersize=16, color="C5", linewidth=3, label='Max Depth: 20')

        ax1.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax2.tick_params(axis="x", labelrotation=90, labelsize=20)
        ax1.tick_params(axis="y", labelsize=20)
        ax1.set_xticks(depth_list)
        ax1.set_xticklabels(depth_list)
        ax2.set_xticks(estimators_list)
        ax2.set_xticklabels(estimators_list)
        ax1.set_xlabel('Max Depth', fontsize=20)
        ax2.set_xlabel('Max Estimators', fontsize=20)
        ax1.set_ylabel('Avg mAP Across Tasks', fontsize=20)
        ax2.set_ylabel('Avg mAP Across Tasks', fontsize=20)

        # handles_1, labels_1 = ax1.get_legend_handles_labels()
        # handles_2, labels_2 = ax1.get_legend_handles_labels()
        # fig.legend(handles_1, labels_1, loc='center right', fontsize=18)
        # fig.legend(handles_2, labels_2, loc='center right', fontsize=18)
        ax1.legend(fontsize=18)
        ax2.legend(fontsize=18)

        plt.show()

        pass

    def plot_gbr(self):
        ig_log_df = pd.DataFrame(np.load(self.path + "ig_log_df_all.npy", allow_pickle=True))
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

        '''
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)
        plt.xticks(rotation=90, fontsize=14)
        ax = sns.violinplot(x="label", y="ig_val", data=ig_log_df)
        ax.set(xlabel='', ylabel='IG Importance')
        plt.show()
        '''

        print("done")

    def plot_r2_celltypes(self):
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

        ax1.plot(pos, r1_hiclstm_gm, marker='o', markersize=16, color='C0', linewidth=3, label='GM12878 (Rao 2014, 3B)')
        ax1.plot(pos, r1_hiclstm_h1, marker='D', markersize=16, color='C1', linewidth=3,
                 label='H1hESC (Dekker 4DN, 2.5B)')
        ax1.plot(pos, r1_hiclstm_wtc, marker='^', markersize=16, color='C2', linewidth=3,
                 label='WTC11 (Dekker 4DN, 1.3B)')
        ax1.plot(pos, r1_hiclstm_hff, marker='v', markersize=16, color='C4', linewidth=3,
                 label='HFFhTERT (Dekker 4DN, 354M)')
        ax1.plot(pos, r1_hiclstm_gmlow, marker='s', markersize=16, color='C3', linewidth=3,
                 label='GM12878 (Aiden 4DN, 300M)')
        ax1.plot(pos, r1_hiclstm_gmlow2, marker='*', markersize=16, color='C5', linewidth=3,
                 label='GM12878 (Aiden 4DN, 216M)')

        ax2.plot(pos, r2_hiclstm_gm, marker='o', markersize=16, color='C0', linewidth=3, label='GM12878 (Rao 2014, 3B)')
        ax2.plot(pos, r2_hiclstm_h1, marker='D', markersize=16, color='C1', linewidth=3,
                 label='H1hESC (Dekker 4DN, 2.5B)')
        ax2.plot(pos, r2_hiclstm_wtc, marker='^', markersize=16, color='C2', linewidth=3,
                 label='WTC11 (Dekker 4DN, 1.3B)')
        ax2.plot(pos, r2_hiclstm_hff, marker='v', markersize=16, color='C4', linewidth=3,
                 label='HFFhTERT (Dekker 4DN, 354M)')
        ax2.plot(pos, r2_hiclstm_gmlow, marker='s', markersize=16, color='C3', linewidth=3,
                 label='GM12878 (Aiden 4DN, 300M)')
        ax2.plot(pos, r2_hiclstm_gmlow2, marker='*', markersize=16, color='C5', linewidth=3,
                 label='GM12878 (Aiden 4DN, 216M)')

        ax1.tick_params(axis="x", labelsize=20, length=0)
        ax2.tick_params(axis="x", labelsize=20, length=0)
        ax1.tick_params(axis="y", labelsize=20)
        ax1.set_xlabel('Distance between positions in Mbp', fontsize=20)
        ax1.set_ylabel('R-squared for Replicate-1', fontsize=20)

        ax2.set_xlabel('Distance between positions in Mbp', fontsize=20)
        ax2.set_ylabel('R-squared for Replicate-2', fontsize=20)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=18)

        plt.show()

        '''
        plt.figure(figsize=(10, 8))
        plt.plot(pos, r1_hiclstm_gm, marker='', markersize=16, color='C0', linewidth=3, label='GM12878 (Rao 2014)')
        plt.plot(pos, r1_hiclstm_h1, marker='o', markersize=16, color='C1', linewidth=3, label='H1hESC (Dekker 4DN)')
        plt.plot(pos, r1_hiclstm_wtc, marker='v', markersize=16, color='C3', linewidth=3, label='WTC11 (Dekker 4DN)')
        plt.plot(pos, r1_hiclstm_gmlow, marker='D', markersize=16, color='C4', linewidth=3,
                 label='GM12878 (low - Aiden 4DN)')
        plt.plot(pos, r1_hiclstm_hff, marker='^', markersize=16, color='C2', linewidth=3, label='HFFhTERT (Dekker 4DN)')

        plt.tick_params(axis="x", labelsize=20, length=0)
        plt.tick_params(axis="y", labelsize=20)
        plt.xlabel('Distance between positions in Mbp', fontsize=20)
        plt.ylabel('R-squared for Replicate-1', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)

        plt.show()
        '''

        pass

    def plot_r2(self, cell):
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

        '''
        plt.figure(figsize=(12, 10))
        plt.plot(pos, r1_hiclstm_full, marker='', markersize=14, color='C0', label='Hi-C-LSTM')
        plt.plot(pos, r1_hiclstm_lstm, marker='o', markersize=14, color='C0', label='Hi-C-LSTM-LSTM')
        plt.plot(pos, r1_hiclstm_cnn, marker='^', markersize=14, color='C0', label='Hi-C-LSTM-CNN')
        plt.plot(pos, r1_hiclstm_fc, marker='v', markersize=14, color='C0', label='Hi-C-LSTM-FC')
        plt.plot(pos, r1_sci_lstm, marker='o', markersize=14, color='C1', label='SCI-LSTM')
        plt.plot(pos, r1_sci_cnn, marker='^', markersize=14, color='C1', label='SCI-CNN')
        plt.plot(pos, r1_sci_fc, marker='v', markersize=14, color='C1', label='SCI-FC')
        plt.plot(pos, r1_sniper_lstm, marker='o', markersize=14, color='C2', label='SNIPER-LSTM')
        plt.plot(pos, r1_sniper_cnn, marker='^', markersize=14, color='C2', label='SNIPER-CNN')
        plt.plot(pos, r1_sniper_fc, marker='v', markersize=14, color='C2', label='SNIPER-FC')
        '''

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))

        ax1.plot(pos, r1_hiclstm_full, marker='o', markersize=16, color='C0', linewidth=3, label='Hi-C-LSTM')
        ax1.plot(pos, r1_hiclstm_lstm, marker='D', markersize=16, color='C0', linewidth=3,
                 linestyle='dashed', label='Hi-C-LSTM-LSTM')
        ax1.plot(pos, r1_hiclstm_cnn, marker='^', markersize=16, color='C0', linewidth=3,
                 linestyle='dotted', label='Hi-C-LSTM-CNN')
        ax1.plot(pos, r1_hiclstm_fc, marker='s', markersize=16, color='C0', linewidth=3,
                 linestyle='dashdot', label='Hi-C-LSTM-FC')
        ax1.plot(pos, r1_sci_lstm, marker='D', markersize=16, color='C1', linewidth=3,
                 linestyle='dashed', label='SCI-LSTM')
        ax1.plot(pos, r1_sci_cnn, marker='^', markersize=16, color='C1', linewidth=3,
                 linestyle='dotted', label='SCI-CNN')
        ax1.plot(pos, r1_sci_fc, marker='s', markersize=16, color='C1', linewidth=3,
                 linestyle='dashdot', label='SCI-FC')
        ax1.plot(pos, r1_sniper_lstm, marker='D', markersize=16, color='C2', linewidth=3,
                 linestyle='dashed', label='SNIPER-LSTM')
        ax1.plot(pos, r1_sniper_cnn, marker='^', markersize=16, color='C2', linewidth=3,
                 linestyle='dotted', label='SNIPER-CNN')
        ax1.plot(pos, r1_sniper_fc, marker='s', markersize=16, color='C2', linewidth=3,
                 linestyle='dashdot', label='SNIPER-FC')

        # ax2.plot(pos, r2_hiclstm_full, marker='', markersize=14, color='C0', label='Hi-C-LSTM')
        ax2.plot(pos, r2_hiclstm_lstm, marker='D', markersize=16, color='C0', linewidth=3,
                 linestyle='dashed', label='Hi-C-LSTM-LSTM')
        ax2.plot(pos, r2_hiclstm_cnn, marker='^', markersize=16, color='C0', linewidth=3,
                 linestyle='dotted', label='Hi-C-LSTM-CNN')
        ax2.plot(pos, r2_hiclstm_fc, marker='s', markersize=16, color='C0', linewidth=3,
                 linestyle='dashdot', label='Hi-C-LSTM-FC')
        ax2.plot(pos, r2_sci_lstm, marker='D', markersize=16, color='C1', linewidth=3,
                 linestyle='dashed', label='SCI-LSTM')
        ax2.plot(pos, r2_sci_cnn, marker='^', markersize=16, color='C1', linewidth=3,
                 linestyle='dotted', label='SCI-CNN')
        ax2.plot(pos, r2_sci_fc, marker='s', markersize=16, color='C1', linewidth=3,
                 linestyle='dashdot', label='SCI-FC')
        ax2.plot(pos, r2_sniper_lstm, marker='D', markersize=16, color='C2', linewidth=3,
                 linestyle='dashed', label='SNIPER-LSTM')
        ax2.plot(pos, r2_sniper_cnn, marker='^', markersize=16, color='C2', linewidth=3,
                 linestyle='dotted', label='SNIPER-CNN')
        ax2.plot(pos, r2_sniper_fc, marker='s', markersize=16, color='C2', linewidth=3,
                 linestyle='dashdot', label='SNIPER-FC')

        ax1.tick_params(axis="x", labelsize=20, length=0)
        ax2.tick_params(axis="x", labelsize=20, length=0)
        ax1.tick_params(axis="y", labelsize=20)
        ax1.set_xlabel('Distance between positions in Mbp', fontsize=20)
        ax1.set_ylabel('R-squared for Replicate-1', fontsize=20)

        ax2.set_xlabel('Distance between positions in Mbp', fontsize=20)
        ax2.set_ylabel('R-squared for Replicate-2', fontsize=20)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=18)

        '''
        plt.tick_params(axis="x", labelsize=20, length=0)
        plt.tick_params(axis="y", labelsize=20)
        plt.xlabel('Distance between positions in Mbp', fontsize=20)
        plt.ylabel('R-squared for Replicate-1', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        '''

        plt.show()

        print("done")

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
        smcko_probs = np.load(self.path + "smcko_probs.npy")

        ctcfko_probs_nl = np.load(self.path + "ctcfko_probs_nl.npy")
        znfko_probs = np.load(self.path + "znfko_probs.npy")
        foxgko_probs = np.load(self.path + "foxgko_probs.npy")
        soxko_probs = np.load(self.path + "soxko_probs.npy")
        xbpko_probs = np.load(self.path + "xbpko_probs.npy")

        # control - KO
        ctcfko_diff = ctcfko_probs - predicted_probs
        convctcf_diff = convctcf_probs - predicted_probs
        divctcf_diff = divctcf_probs - predicted_probs
        radko_diff = radko_probs - predicted_probs
        smcko_diff = smcko_probs - predicted_probs

        ctcfnl_diff = ctcfko_probs_nl - predicted_probs
        znfko_diff = znfko_probs - predicted_probs
        foxgko_diff = foxgko_probs - predicted_probs
        soxko_diff = soxko_probs - predicted_probs
        xbpko_diff = xbpko_probs - predicted_probs

        df_main = pd.DataFrame(columns=["pos", "CTCF_Cohesin_KO_Loop", "Convergent_CTCF", "Divergent_CTCF",
                                        "CTCF_KO_nl", "ZNF143_KO", "FOXG1_KO", "SOX2_KO", "XBP1_KO"])
        df_main["pos"] = pos
        # df_main["No KO"] = predicted_probs
        df_main["CTCF_Cohesin_KO_Loop"] = ctcfko_diff
        df_main["Convergent_CTCF"] = convctcf_diff
        df_main["Divergent_CTCF"] = divctcf_diff
        df_main["CTCF_KO_nl"] = ctcfnl_diff
        df_main["ZNF143_KO"] = znfko_diff
        df_main["FOXG1_KO"] = foxgko_diff
        df_main["SOX2_KO"] = soxko_diff
        df_main["XBP1_KO"] = xbpko_diff

        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Distance between positions in Mbp", fontsize=20)
        plt.ylabel("Average Difference in Contact Strength \n (KO - No KO)", fontsize=20)
        # plt.plot('pos', 'No KO', data=df_main, marker='o', markersize=14, color="C3", linewidth=2, label="No KO")
        plt.plot('pos', 'CTCF_Cohesin_KO_Loop', data=df_main, marker='o', markersize=16, color="C0", linewidth=3, label="CTCF+Cohesin KO (Loop)")
        plt.plot('pos', 'Convergent_CTCF', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 linestyle='dotted', label="Div->Conv CTCF")
        plt.plot('pos', 'Divergent_CTCF', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 linestyle='dashed', label="Conv->Div CTCF")
        plt.plot('pos', 'CTCF_KO_nl', data=df_main, marker='s', markersize=16, color="C2", linewidth=3,
                 linestyle='dotted', label="CTCF KO (Non-loop)")
        plt.plot('pos', 'ZNF143_KO', data=df_main, marker='^', markersize=16, color="C3", linewidth=3,
                 linestyle='dashed', label="ZNF143 KO")
        plt.plot('pos', 'FOXG1_KO', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 linestyle='dashdot', label="FOXG1 KO")
        plt.plot('pos', 'SOX2_KO', data=df_main, marker='x', markersize=16, color="C6", linewidth=3, label="SOX2 KO")
        plt.plot('pos', 'XBP1_KO', data=df_main, marker='+', markersize=16, color="C7", linewidth=3, label="XBP1 KO")

        #plt.plot('pos', 'RAD21 KO', data=df_main, marker='s', markersize=16, color="C2", linewidth=3,
        #         linestyle='dashdot', label="RAD21 KO")
        #plt.plot('pos', 'SMC3 KO', data=df_main, marker='^', markersize=16, color="C4", linewidth=3, label="SMC3 KO")
        plt.legend(fontsize=18)
        plt.show()

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
        loop_p = np.load(path + "loops/" + precision_file)
        loop_r = np.load(path + "loops/" + recall_file)
        subc_p = np.load(path + "subcompartments/" + precision_file)
        subc_r = np.load(path + "subcompartments/" + recall_file)

        # list(map(lambda x: 10 if x == 'x' else x, a))

        plt.figure(figsize=(8, 6))
        plt.step(gene_r, gene_p, color='b', alpha=0.5, where='post', linewidth=3, label="Gene Expression")
        plt.step(rep_r, rep_p, color='g', alpha=0.5, where='post', linewidth=3, label="Replication Timing")
        plt.step(en_r, en_p, color='r', alpha=0.5, where='post', linewidth=3, label="Enhancers")
        plt.step(tss_r, tss_p, color='c', alpha=0.5, where='post', linewidth=3, label="TSS")
        plt.step(pe_r, pe_p, color='m', alpha=0.5, where='post', linewidth=3, label="PE-Interactions")
        plt.step(fire_r, fire_p, color='y', alpha=0.5, where='post', linewidth=3, label="FIREs")
        plt.step(domain_r, domain_p, color='k', alpha=0.5, where='post', linewidth=3, label="Non-loop Domains")
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

        mode = "chr21"

        if mode == "tad":
            pos = np.arange(-110, 120, 10)
            plt.figure(figsize=(6, 2))
            feature_signal = np.load(self.path + "lstm/" + "feature_signal_tad.npy")
            plt.plot(pos, feature_signal)
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
    cfg = config.Config()
    plot_ob = PlotFns(cfg)

    #plot_ob.plot_combined(cell = "GM12878")
    # plot_ob.plot_combined_all(cell="GM12878")
    # plot_ob.plot_mAP_celltypes()
    # plot_ob.plot_auroc_celltypes()
    # plot_ob.plot_auroc()

    # hidden_list = [4, 8, 16, 32, 64, 128]
    # plot_ob.plot_hidden(hidden_list)

    # plot_ob.plot_xgb()
    # plot_ob.plot_gbr()

    # plot_ob.plot_r2(cell = "H1hESC")
    # plot_ob.plot_r2_celltypes()
    # plot_ob.plot_symmetry()

    plot_ob.plot_knockout_results()
    # plot_ob.plot_knockout_tfs()
    # plot_ob.pr_curves()

    # plot_ob.plot_feature_signal()
    # plot_ob.plot_pred_range()

    '''
    chr = 21
    pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")
    hic_mat, st = plot_ob.plot_heatmaps(pred_data)
    plot_ob.ctcf_dots(hic_mat, st, chr)
    '''

    print("done")
