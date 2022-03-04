import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sn
import training.config as config
import matplotlib.pyplot as plt


def get_heatmaps(data, no_pred=False):
    st = int(data["i"].min())
    data["i"] = data["i"] - st
    data["j"] = data["j"] - st
    nr = int(data["j"].max()) + 1
    rows = np.array(data["i"]).astype(int)
    cols = np.array(data["j"]).astype(int)

    hic_mat = np.zeros((nr, nr))
    hic_mat[rows, cols] = np.array(data["v"])
    hic_upper = np.triu(hic_mat)
    if no_pred:
        hic_mat[cols, rows] = np.array(data["v"])
    else:
        hic_mat[cols, rows] = np.array(data["pred"])

    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2

    return hic_mat, st


def plot_foxg1(data):
    site = 222863
    data["i"] = data["i"] - site
    data["j"] = data["j"] - site

    data = data.loc[(data["i"] >= -100) & (data["i"] <= 100) &
                    (data["j"] >= -100) & (data["j"] <= 100)]

    data["i"] = data["i"] + 100
    data["j"] = data["j"] + 100

    nr = 201
    rows = np.array(data["i"]).astype(int)
    cols = np.array(data["j"]).astype(int)
    hic_mat = np.zeros((nr, nr))
    hic_mat[rows, cols] = np.array(data["v"])
    hic_upper = np.triu(hic_mat)
    hic_mat[cols, rows] = np.array(data["pred"])
    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2

    simple_plot(hic_mat)
    pass


def plot_tal1_lmo2(data):
    tal_data = data.loc[data["i"] < 5000]
    lmo2_data = data.loc[data["i"] > 5000]

    get_heatmaps(tal_data)
    get_heatmaps(lmo2_data)
    pass


def simple_plot(hic_win):
    '''
    plt.imshow(hic_win, cmap='hot', interpolation='nearest')
    plt.yticks([])
    plt.xticks([])
    plt.show()
    '''

    plt.figure()
    sns.set_theme()
    ax = sns.heatmap(hic_win, cmap="Reds", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig("/home/kevindsouza/Downloads/ctcf_ko.png")
    #plt.show()

    '''
    sns.set_theme()
    rdgn = sns.diverging_palette(h_neg=220, h_pos=14, s=79, l=55, sep=3, as_cmap=True)
    sns.heatmap(hic_win, cmap=rdgn, center=0.00, cbar=True)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    '''

    pass


def plot_frame_error(error_list):
    pos_list = np.arange(0, 150)
    plt.figure()
    # plt.title("Average Prediction Error within Frame")
    plt.xlabel("Position in Frame", fontsize=14)
    plt.ylabel("Average Error", fontsize=14)
    plt.plot(pos_list, error_list)
    plt.grid(False)
    plt.show()
    pass


def plot_smoothness(embeddings):
    window = 2000
    nrows = len(embeddings)
    diff_list = np.arange(-window, window + 1)
    diff_list = np.delete(diff_list, [window])
    diff_vals = np.zeros((nrows, 2 * window))
    for r in range(nrows):
        for i, d in enumerate(diff_list):
            if (r + d) >= 0 and (r + d) <= nrows - 1:
                diff_vals[r, i] = np.linalg.norm(embeddings[r, :] - embeddings[r + d, :], ord=1)
            else:
                continue

    diff_reduce = diff_vals.mean(axis=0)
    plt.title("Average L2 Norm of Embeddings with Distance")
    plt.xlabel("Distance in 10 Kbp", fontsize=14)
    plt.ylabel("Average L2 Norm", fontsize=14)
    plt.plot(diff_list, diff_reduce)
    plt.grid(b=None)
    plt.show()
    pass


def plot3d(embeddings):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 'red')
    plt.show()
    pass


def plot_euclid_heatmap(embeddings):
    nr = len(embeddings)
    euclid_heatmap = np.zeros((nr, nr))

    for r1 in range(nr):
        for r2 in range(nr):
            euclid_heatmap[r1, r2] = np.linalg.norm(embeddings[r1, :] - embeddings[r2, :])

    simple_plot(euclid_heatmap)
    pass


def plot_pr_curve(precision, recall):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.savefig('XGBoost_PR')
    plt.show()


def plot_confusion_matrix(predictions):
    conf_matrix = confusion_matrix(predictions[:, 7], predictions[:, 6])
    conf_matrix = conf_matrix[1:, 1:]
    df_cm = pd.DataFrame(conf_matrix)
    df_cm = df_cm.div(df_cm.sum(axis=0), axis=1)

    x_axis_labels = ["A2", "A1", "B1", "B2", "B3"]
    y_axis_labels = ["A2",
                     "A1", "B1", "B2", "B3"]

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", xticklabels=x_axis_labels,
               yticklabels=y_axis_labels)  # font size

    plt.show()


def plot_combined(map_frame):
    tasks = ["Gene Expression", "Replication Timing", "Enhancers", "TSS", "PE-Interactions", "FIREs",
             "Non-loop Domains", "Loop Domains"]

    df_main = pd.DataFrame(columns=["Tasks", "Hi-C-LSTM"])
    df_main["Tasks"] = tasks
    df_main["Hi-C-LSTM"] = [map_frame["gene_map"].mean(), map_frame["rep_map"].mean(),
                            map_frame["enhancers_map"].mean(), map_frame["tss_map"].mean(),
                            map_frame["pe_map"].mean(), map_frame["fire_map"].mean(),
                            map_frame["domains_map"].mean(), map_frame["loops_map"].mean()]

    plt.figure(figsize=(12, 10))
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Prediction Target", fontsize=20)
    plt.ylabel("mAP ", fontsize=20)
    plt.plot('Tasks', 'Hi-C-LSTM', data=df_main, marker='o', markersize=16, color="C3",
             linewidth=3,
             label="Hi-C-LSTM")
    plt.legend(fontsize=18)
    plt.show()

    pass


def plot_gbr(main_df):
    """
    captum_test(cfg, model, chr) -> DataFrame
    Gets data for chromosome and cell type. Runs IG using captum.
    Args:
        cfg (Config): The configuration to use for the experiment.
        model (SeqLSTM): The model to run captum on.
        chr (int): The chromosome to run captum on.
    """
    main_df["ig"] = main_df["ig"].astype(float)

    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.8)
    sns.set_style(style='white')
    plt.xticks(rotation=90, fontsize=20)
    plt.ylim(-1, 1)
    ax = sns.violinplot(x="target", y="ig", data=main_df)
    ax.set(xlabel='', ylabel='IG Importance')
    plt.show()


def plot_r2(comb_r2_df):
    max_diff = int(comb_r2_df['diff'].max())
    max_mb = 50
    pos = [10, 20, 30, 40, 50]
    avg_diff = pd.DataFrame(columns=["diff", "r2"])
    r2_list = []
    r2_list_pos = []

    for diff in range(max_diff):
        subset_diff = comb_r2_df.loc[comb_r2_df["diff"] == diff]
        r2_mean = subset_diff["r2"].mean()
        avg_diff = avg_diff.append({"diff": diff, "r2": r2_mean}, ignore_index=True)

    for i in range(max_mb):
        r2_sub = avg_diff.loc[i:(i + 1) * 100, :]
        r2_mean = r2_sub["r2"].mean(skipna=True)
        r2_list.append(r2_mean)

    for k in range(5):
        r2_list_pos.append(np.mean(r2_list[k * 10: (k + 1) * 10]))

    plt.figure(figsize=(12, 10))
    plt.plot(pos, r2_list_pos, marker='', markersize=14, color='C0', label='Hi-C-LSTM')
    plt.tick_params(axis="x", labelsize=20, length=0)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel('Distance between positions in Mbp', fontsize=20)
    plt.ylabel('R-squared for Replicate-1', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()
    pass


def scatter_tal_lm(diff_mat):
    pred = np.tril(diff_mat)
    pred = pred.flatten(order='C')
    pred_nz = pred[pred != 0]

    og = np.triu(diff_mat)
    og = og.flatten(order='F')
    og_nz = og[og != 0]

    plt.scatter(og_nz, pred_nz, marker='o')
    plt.tick_params(axis="x", labelsize=20, length=0)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel('(KO - WT) Original', fontsize=20)
    plt.ylabel('(KO - WT) Predicted', fontsize=20)
    plt.show()
    print("done")


def hist_2d(og, pred):
    x_min = np.min(og)
    x_max = np.max(og)

    y_min = np.min(pred)
    y_max = np.max(pred)

    x_bins = np.linspace(x_min, x_max, 50)
    y_bins = np.linspace(y_min, y_max, 50)

    # Creating plot
    plt.figure(figsize=(10, 8))
    hist, _, _, _ = plt.hist2d(og, pred, bins=[x_bins, y_bins])

    plt.xticks(fontsize=18)
    plt.xlim([0, 0.1])
    plt.yticks(fontsize=18)
    plt.ylim([0.004, 0.1])
    plt.xlabel('LMO2 KO - WT (Original)', fontsize=20)
    plt.ylabel('LMO2 KO - WT (Predicted)', fontsize=20)

    # show plot
    plt.tight_layout()
    plt.savefig("/home/kevindsouza/Downloads/lmo2_hist.png")
    pass


def barplot_tal_lm():
    tal_og = np.load(cfg.output_directory + "tal1og_difflist.npy")
    tal_pred = np.load(cfg.output_directory + "tal1pred_difflist.npy")

    lmo2_og = np.load(cfg.output_directory + "lmo2og_difflist.npy")
    lmo2_pred = np.load(cfg.output_directory + "lmo2pred_difflist.npy")

    hist_2d(lmo2_og, lmo2_pred)

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.8)
    sns.set_style(style='white')

    data_lists = [tal_og, tal_pred, lmo2_og, lmo2_pred]
    label_lists = ["TAL1 Original (KO - WT)", "TAL1 Predicted (KO - WT)", "LMO2 Original (KO - WT)",
                   "LMO2 Predicted (KO - WT)"]
    tallm_df = pd.DataFrame(columns=["data", "label"])
    for i, l in enumerate(label_lists):
        temp = pd.DataFrame(columns=["data", "label"])
        temp["data"] = data_lists[i]
        temp["label"] = l
        tallm_df = tallm_df.append(temp)

    plt.figure(figsize=(14, 12))
    sns.barplot(x="label", y="data", data=tallm_df, ci="sd")
    plt.xticks(rotation=90)
    plt.xlabel('Data', fontsize=20)
    plt.ylabel('Contact Strengths', fontsize=20)
    plt.subplots_adjust(bottom=0.4)
    # plt.show()
    plt.savefig("/home/kevindsouza/Downloads/bar_tal_lm.png")
    print("done")


if __name__ == '__main__':
    plot_chr = list(range(14, 15))
    cfg = config.Config()
    cell = cfg.cell
    comb_r2_df = pd.DataFrame(columns=["diff", "r2"])

    for chr in plot_chr:
        '''
        r2_diff = pd.read_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")
        r2_diff = r2_diff.drop(['Unnamed: 0'], axis=1)
        comb_r2_df = comb_r2_df.append(r2_diff, ignore_index=True)
        plot_r2(comb_r2_df)
        '''

        # pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        # plot_foxg1(pred_data)
        '''
        foxg1_data = pd.read_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (cell, str(chr)), sep="\t")
        plot_foxg1(foxg1_data)
        '''

        # foxg1_ko = np.load(cfg.output_directory + "foxg1_ko.npy")
        # simple_plot(foxg1_ko)

        # tal1_diff = np.load(cfg.output_directory + "tal1_diff.npy")
        # simple_plot(tal1_diff)
        # scatter_tal_lm(tal1_diff)
        barplot_tal_lm()

        # pred_data = pd.read_csv(cfg.output_directory + "%s_predictions_chr.csv" % (cell), sep="\t")
        # plot_tal1_lmo2(pred_data)

        # tal_ko = pd.read_csv(cfg.hic_path + cell +"/talko_tal_df.txt", sep="\t")
        # plot_heatmaps(tal_ko)

        # pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        # plot_heatmaps(pred_data)

    print("done")

    # foxg1_data = pd.read_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (self.cfg.cell, str(self.chr)), sep="\t")
    # pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
    # pred_data = pd.read_csv(cfg.output_directory + "testwin_predictions_chr%s.csv" % str(chr), sep="\t")
    # pred_data = pd.read_csv(cfg.output_directory + "combined150_melobfko_chr%s.csv" % str(chr), sep="\t")
    # pred_data = pd.read_csv(cfg.output_directory + "combined150_meloafkofusion_chr%s.csv" % str(chr), sep="\t")
    # embeddings = np.load(cfg.output_directory + "combined_150_embeddings_chr21.npy")
    # Whic_win_af = np.load(cfg.output_directory + "hic_melowin_afko2.npy")
    # hic_win_bf = np.load(cfg.output_directory + "hic_melowin.npy")
    # error_list = np.load(cfg.output_directory + "combined150_frameerror_chr%s.npy" % str(chr))

    # simple_plot(hic_win_bf)
    # plot_heatmaps(pred_data)
    # plot_smoothness(embeddings)
    # plot3d(embeddings)
    # plot_euclid_heatmap(embeddings)
    # plot_frame_error(error_list)
