import logging
import numpy as np
import matplotlib.pyplot as plt
import operator
import pandas as pd
import seaborn as sns
import training.config as config

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

    def plot_combined(self, cell):
        tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
                 "Non-loop Domains",
                 "Loop Domains", "Subcompartments"]

        if cell == "GM12878":
            lstm_values_all_tasks = np.load(self.path + "lstm/" + "lstm_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "lstm/" + "sniper_intra_values_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "lstm/" + "sniper_inter_values_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "lstm/" + "graph_values_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "lstm/" + "pca_values_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "lstm/" + "sbcid_values_all_tasks.npy")
        if cell == "H1hESC":
            lstm_values_all_tasks = np.load(self.path + "h1_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_values_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_values_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_values_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_values_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_values_all_tasks.npy")
        if cell == "HFFhTERT":
            lstm_values_all_tasks = np.load(self.path + "hff_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_values_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_values_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_values_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_values_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_values_all_tasks.npy")
        if cell == "WTC11":
            lstm_values_all_tasks = np.load(self.path + "wtc_values_all_tasks.npy")
            sniper_intra_values_all_tasks = np.load(self.path + "sniper_intra_values_all_tasks.npy")
            sniper_inter_values_all_tasks = np.load(self.path + "sniper_inter_values_all_tasks.npy")
            graph_values_all_tasks = np.load(self.path + "graph_values_all_tasks.npy")
            pca_values_all_tasks = np.load(self.path + "pca_values_all_tasks.npy")
            sbcid_values_all_tasks = np.load(self.path + "sbcid_values_all_tasks.npy")

        df_main = pd.DataFrame(columns=["Tasks", "Hi-C-LSTM", "SNIPER-INTRA", "SNIPER-INTER", "SCI", "PCA", "SBCID"])
        df_main["Tasks"] = tasks
        df_main["Hi-C-LSTM"] = lstm_values_all_tasks
        df_main["SNIPER-INTRA"] = sniper_intra_values_all_tasks
        df_main["SNIPER-INTER"] = sniper_inter_values_all_tasks
        df_main["SCI"] = graph_values_all_tasks
        df_main["PCA"] = pca_values_all_tasks
        df_main["SBCID"] = sbcid_values_all_tasks

        palette = {"Hi-C-LSTM": "C3", "SNIPER-INTRA": "C0", "SNIPER-INTER": "C1", "SCI": "C2", "PCA": "C4",
                   "SBCID": "C5"}
        plt.figure(figsize=(12, 10))
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
        gmlow_values_all_tasks = hff_values_all_tasks + 0.01

        df_main = pd.DataFrame(columns=["Tasks", "GM12878 (Rao 2014)", "H1hESC (Dekker 4DN)", "WTC11 (Dekker 4DN)",
                                        "GM12878 (low - Aiden 4DN)", "HFFhTERT (Dekker 4DN)"])
        df_main["Tasks"] = tasks
        df_main["GM12878 (Rao 2014)"] = gm_values_all_tasks
        df_main["H1hESC (Dekker 4DN)"] = h1_values_all_tasks
        df_main["WTC11 (Dekker 4DN)"] = wtc_values_all_tasks
        df_main["GM12878 (low - Aiden 4DN)"] = gmlow_values_all_tasks
        df_main["HFFhTERT (Dekker 4DN)"] = hff_values_all_tasks

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("mAP ", fontsize=20)
        plt.plot('Tasks', 'GM12878 (Rao 2014)', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="GM12878 (Rao 2014)")
        plt.plot('Tasks', 'H1hESC (Dekker 4DN)', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 label="H1hESC (Dekker 4DN)")
        plt.plot('Tasks', 'WTC11 (Dekker 4DN)', data=df_main, marker='^', markersize=16, color="C2", linewidth=3,
                 label="WTC11 (Dekker 4DN)")
        plt.plot('Tasks', 'GM12878 (low - Aiden 4DN)', data=df_main, marker='s', markersize=16, color="C3", linewidth=3,
                 label="GM12878 (low - Aiden 4DN)")
        plt.plot('Tasks', 'HFFhTERT (Dekker 4DN)', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 label="HFFhTERT (Dekker 4DN)")
        plt.legend(fontsize=18)
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

        df_main = pd.DataFrame(columns=["Tasks", "GM12878 (Rao 2014)", "H1hESC (Dekker 4DN)", "WTC11 (Dekker 4DN)",
                                        "GM12878 (low - Aiden 4DN)", "HFFhTERT (Dekker 4DN)"])
        df_main["Tasks"] = tasks
        df_main["GM12878 (Rao 2014)"] = gm_auroc_all_tasks
        df_main["H1hESC (Dekker 4DN)"] = h1_auroc_all_tasks
        df_main["WTC11 (Dekker 4DN)"] = wtc_auroc_all_tasks
        df_main["GM12878 (low - Aiden 4DN)"] = gmlow_auroc_all_tasks
        df_main["HFFhTERT (Dekker 4DN)"] = hff_auroc_all_tasks

        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Prediction Target", fontsize=20)
        plt.ylabel("AuROC ", fontsize=20)
        plt.plot('Tasks', 'GM12878 (Rao 2014)', data=df_main, marker='o', markersize=16, color="C0", linewidth=3,
                 label="GM12878 (Rao 2014)")
        plt.plot('Tasks', 'H1hESC (Dekker 4DN)', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 label="H1hESC (Dekker 4DN)")
        plt.plot('Tasks', 'WTC11 (Dekker 4DN)', data=df_main, marker='^', markersize=16, color="C2", linewidth=3,
                 label="WTC11 (Dekker 4DN)")
        plt.plot('Tasks', 'GM12878 (low - Aiden 4DN)', data=df_main, marker='s', markersize=16, color="C3", linewidth=3,
                 label="GM12878 (low - Aiden 4DN)")
        plt.plot('Tasks', 'HFFhTERT (Dekker 4DN)', data=df_main, marker='v', markersize=16, color="C4", linewidth=3,
                 label="HFFhTERT (Dekker 4DN)")

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

        r2_hiclstm_gm = np.load(self.path + "r2_hiclstm_lstm.npy")
        r2_hiclstm_h1 = np.load(self.path + "r2_sci_cnn.npy")
        r2_hiclstm_wtc = np.load(self.path + "r2_sniper_cnn.npy")
        r2_hiclstm_gmlow = np.load(self.path + "r2_sci_fc.npy")
        r2_hiclstm_hff = np.load(self.path + "r2_sniper_fc.npy")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))

        ax1.plot(pos, r1_hiclstm_gm, marker='o', markersize=16, color='C0', linewidth=3, label='GM12878 (Rao 2014)')
        ax1.plot(pos, r1_hiclstm_h1, marker='D', markersize=16, color='C1', linewidth=3, label='H1hESC (Dekker 4DN)')
        ax1.plot(pos, r1_hiclstm_wtc, marker='^', markersize=16, color='C2', linewidth=3, label='WTC11 (Dekker 4DN)')
        ax1.plot(pos, r1_hiclstm_gmlow, marker='s', markersize=16, color='C3', linewidth=3,
                 label='GM12878 (low - Aiden 4DN)')
        ax1.plot(pos, r1_hiclstm_hff, marker='v', markersize=16, color='C4', linewidth=3,
                 label='HFFhTERT (Dekker 4DN)')

        ax2.plot(pos, r2_hiclstm_gm, marker='o', markersize=16, color='C0', linewidth=3, label='GM12878 (Rao 2014)')
        ax2.plot(pos, r2_hiclstm_h1, marker='D', markersize=16, color='C1', linewidth=3, label='H1hESC (Dekker 4DN)')
        ax2.plot(pos, r2_hiclstm_wtc, marker='^', markersize=16, color='C2', linewidth=3, label='WTC11 (Dekker 4DN)')
        ax2.plot(pos, r2_hiclstm_gmlow, marker='s', markersize=16, color='C3', linewidth=3,
                 label='GM12878 (low - Aiden 4DN)')
        ax2.plot(pos, r2_hiclstm_hff, marker='v', markersize=16, color='C4', linewidth=3,
                 label='HFFhTERT (Dekker 4DN)')

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

    def plot_r2(self):
        r1_hiclstm_full = np.load(self.path + "lstm/" + "r1_hiclstm_full.npy")
        r1_hiclstm_lstm = np.load(self.path + "lstm/" + "r1_hiclstm_lstm.npy")
        r1_hiclstm_cnn = np.load(self.path + "lstm/" + "r1_hiclstm_cnn.npy")
        r1_sci_lstm = np.load(self.path + "lstm/" + "r1_sci_lstm.npy")
        r1_sniper_lstm = np.load(self.path + "lstm/" + "r1_sniper_lstm.npy")
        r1_sci_cnn = np.load(self.path + "lstm/" + "r1_sci_cnn.npy")
        r1_sniper_cnn = np.load(self.path + "lstm/" + "r1_sniper_cnn.npy")
        r1_hiclstm_fc = np.load(self.path + "lstm/" + "r1_hiclstm_fc.npy")
        r1_sci_fc = np.load(self.path + "lstm/" + "r1_sci_fc.npy")
        r1_sniper_fc = np.load(self.path + "lstm/" + "r1_sniper_fc.npy")

        r2_hiclstm_lstm = np.load(self.path + "lstm/" + "r2_hiclstm_lstm.npy")
        r2_hiclstm_cnn = np.load(self.path + "lstm/" + "r2_hiclstm_cnn.npy")
        r2_sci_lstm = np.load(self.path + "lstm/" + "r2_sci_lstm.npy")
        r2_sniper_lstm = np.load(self.path + "lstm/" + "r2_sniper_lstm.npy")
        r2_sci_cnn = np.load(self.path + "lstm/" + "r2_sci_cnn.npy")
        r2_sniper_cnn = np.load(self.path + "lstm/" + "r2_sniper_cnn.npy")
        r2_hiclstm_fc = np.load(self.path + "lstm/" + "r2_hiclstm_fc.npy")
        r2_sci_fc = np.load(self.path + "lstm/" + "r2_sci_fc.npy")
        r2_sniper_fc = np.load(self.path + "lstm/" + "r2_sniper_fc.npy")
        # r2_hiclstm_full = np.load(self.path + "lstm/" + "r2_hiclstm_full.npy")

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

    def plot_knockout_results(self):
        pos = np.linspace(0, 1, 11)
        predicted_probs = np.load(cfg.hic_path + "GM12878/" + "predicted_probs.npy")
        ctcfko_probs = np.load(cfg.hic_path + "GM12878/" + "ctcfko_probs.npy")
        convctcf_probs = np.load(cfg.hic_path + "GM12878/" + "convctcf_probs.npy")
        divctcf_probs = np.load(cfg.hic_path + "GM12878/" + "divctcf_probs.npy")
        radko_probs = np.load(cfg.hic_path + "GM12878/" + "radko_probs.npy")
        smcko_probs = np.load(cfg.hic_path + "GM12878/" + "smcko_probs.npy")

        # control - KO
        ctcfko_diff = ctcfko_probs - predicted_probs
        convctcf_diff = convctcf_probs - predicted_probs
        divctcf_diff = divctcf_probs - predicted_probs
        radko_diff = radko_probs - predicted_probs
        smcko_diff = smcko_probs - predicted_probs

        df_main = pd.DataFrame(columns=["pos", "CTCF KO", "Convergent CTCF", "Divergent CTCF", "RAD21 KO", "SMC3 KO"])
        df_main["pos"] = pos
        # df_main["No KO"] = predicted_probs
        df_main["CTCF KO"] = ctcfko_diff
        df_main["Convergent CTCF"] = convctcf_diff
        df_main["Divergent CTCF"] = divctcf_diff
        df_main["RAD21 KO"] = radko_diff
        df_main["SMC3 KO"] = smcko_diff

        palette = {"CTCF KO": "C0", "Convergent CTCF": "C5", "Divergent CTCF": "C1", "RAD21 KO": "C2",
                   "SMC3 KO": "C4"}
        plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Distance between positions in Mbp", fontsize=20)
        plt.ylabel("Average Difference in Contact Strength \n (KO - No KO)", fontsize=20)
        # plt.plot('pos', 'No KO', data=df_main, marker='o', markersize=14, color="C3", linewidth=2, label="No KO")
        plt.plot('pos', 'CTCF KO', data=df_main, marker='o', markersize=16, color="C0", linewidth=3, label="CTCF KO")
        plt.plot('pos', 'Convergent CTCF', data=df_main, marker='*', markersize=16, color="C5", linewidth=3,
                 linestyle='dotted', label="Div->Conv CTCF")
        plt.plot('pos', 'Divergent CTCF', data=df_main, marker='D', markersize=16, color="C1", linewidth=3,
                 linestyle='dashed', label="Conv->Div CTCF")
        plt.plot('pos', 'RAD21 KO', data=df_main, marker='s', markersize=16, color="C2", linewidth=3,
                 linestyle='dashdot', label="RAD21 KO")
        plt.plot('pos', 'SMC3 KO', data=df_main, marker='^', markersize=16, color="C4", linewidth=3, label="SMC3 KO")
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

    # plot_ob.plot_combined(cell = "H1hESC")
    # plot_ob.plot_mAP_celltypes()
    # plot_ob.plot_auroc_celltypes()
    # plot_ob.plot_auroc()

    # hidden_list = [4, 8, 16, 32, 64, 128]
    # plot_ob.plot_hidden(hidden_list)

    # plot_ob.plot_xgb()
    # plot_ob.plot_gbr()

    # plot_ob.plot_r2()
    # plot_ob.plot_r2_celltypes()
    # plot_ob.plot_symmetry()

    # plot_ob.plot_knockout_results()
    # plot_ob.pr_curves()

    # plot_ob.plot_feature_signal()
    # plot_ob.plot_pred_range()

    print("done")
