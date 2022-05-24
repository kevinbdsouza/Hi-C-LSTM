import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sn
from training.config import Config
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import curve_fit


def get_heatmaps(data, no_pred=False):
    """
    get_heatmaps(data, no_pred) -> Array, int
    Gets upper and lower comperison heatmaps.
    Args:
        data (DataFrame): Frame with values and predictions
        no_pred (bool): One of True or False. If True, then plots observed values on both sides.
    """

    st = int(data["i"].min())
    data["i"] = data["i"] - st
    data["j"] = data["j"] - st
    nr = int(data["j"].max()) + 1
    rows = np.array(data["i"]).astype(int)
    cols = np.array(data["j"]).astype(int)

    "initialize"
    hic_mat = np.zeros((nr, nr))
    hic_mat[rows, cols] = np.array(data["v"])
    hic_upper = np.triu(hic_mat)

    "check for pred"
    if no_pred:
        hic_mat[cols, rows] = np.array(data["v"])
    else:
        hic_mat[cols, rows] = np.array(data["pred"])

    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2
    return hic_mat, st


def plot_foxg1(cfg, data):
    """
    plot_foxg1(cfg, data) -> No return object
    Plots window around foxg1 ko site.
    Args:
        cfg (Config): configuration to use
        data (DataFrame): Frame with values and predictions
    """

    site = cfg.foxg1_indices
    data["i"] = data["i"] - site
    data["j"] = data["j"] - site

    "window"
    data = data.loc[(data["i"] >= -100) & (data["i"] <= 100) &
                    (data["j"] >= -100) & (data["j"] <= 100)]
    data["i"] = data["i"] + 100
    data["j"] = data["j"] + 100

    "form matrix"
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

    "plot"
    simple_plot(hic_mat, mode="reds")


def simple_plot(hic_win, mode):
    """
    simple_plot(hic_win, mode) -> No return object
    plots heatmaps of reds or differences.
    Args:
        hic_win (Array): Matrix of Hi-C values
        mode (string): one of reds or diff
    """

    if mode == "reds":
        plt.figure()
        sns.set_theme()
        ax = sns.heatmap(hic_win, cmap="Reds", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([])
        # plt.savefig("/home/kevindsouza/Downloads/ctcf_ko.png")
        plt.show()

    if mode == "diff":
        plt.figure()
        sns.set_theme()
        rdgn = sns.diverging_palette(h_neg=220, h_pos=14, s=79, l=55, sep=3, as_cmap=True)
        sns.heatmap(hic_win, cmap=rdgn, center=0.00, cbar=True)
        plt.yticks([])
        plt.xticks([])
        # plt.savefig("/home/kevindsouza/Downloads/ctcf_ko.png")
        plt.show()


def indices_diff_mat(indice, st, hic_mat, mode="ctcf"):
    """
    indices_diff_mat(indice, st, hic_mat, mode) -> Array
    gets window matrices given indices
    Args:
        indice (Array): Matrix of Hi-C values
        st (int): Starting indice
        hic_mat (Array): Matrix of Hi-C values
        mode (string): tadbs or others
    """

    nrows = len(hic_mat)

    if mode == "tadbs":
        i = indice[0] - st
        j = indice[1] - st
        if i - 98 >= 0:
            win_start = i - 98
        else:
            win_start = 0
        if j + 98 <= (nrows - 1):
            win_stop = i + 98
        else:
            win_stop = nrows - 1
    else:
        i = indice - st
        if i - 100 >= 0:
            win_start = i - 100
        else:
            win_start = 0
        if i + 100 <= (nrows - 1):
            win_stop = i + 100
        else:
            win_stop = nrows - 1

    hic_win = hic_mat[win_start:win_stop, win_start:win_stop]
    return hic_win


def plot_frame_error(error_list):
    """
    plot_frame_error(error_list) -> No return object
    Plot frame error given error list
    Args:
        error_list (List): List of errors
    """

    pos_list = np.arange(0, 150)
    plt.figure()
    plt.xlabel("Position in Frame", fontsize=14)
    plt.ylabel("Average Error", fontsize=14)
    plt.plot(pos_list, error_list)
    plt.grid(False)
    plt.show()


def plot_smoothness(representations):
    """
    plot_smoothness(representations) -> No return object
    Plot smoothness of representations.
    Args:
        representations (Array): representation matrix
    """

    window = 2000
    nrows = len(representations)
    diff_list = np.arange(-window, window + 1)
    diff_list = np.delete(diff_list, [window])
    diff_vals = np.zeros((nrows, 2 * window))
    for r in range(nrows):
        for i, d in enumerate(diff_list):
            if (r + d) >= 0 and (r + d) <= nrows - 1:
                diff_vals[r, i] = np.linalg.norm(representations[r, :] - representations[r + d, :], ord=1)
            else:
                continue

    diff_reduce = diff_vals.mean(axis=0)
    plt.title("Average L2 Norm of Embeddings with Distance")
    plt.xlabel("Distance in 10 Kbp", fontsize=14)
    plt.ylabel("Average L2 Norm", fontsize=14)
    plt.plot(diff_list, diff_reduce)
    plt.grid(b=None)
    plt.show()


def plot3d(representations):
    """
    plot3d(representations) -> No return object
    Plot first 3 dims of representations.
    Args:
        representations (Array): representation matrix
    """

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(representations[:, 0], representations[:, 1], representations[:, 2], 'red')
    plt.show()


def plot_euclid_heatmap(representations):
    """
    plot_euclid_heatmap(representations) -> No return object
    Plot heatmap of euclidean distance.
    Args:
        representations (Array): representation matrix
    """

    nr = len(representations)
    euclid_heatmap = np.zeros((nr, nr))

    for r1 in range(nr):
        for r2 in range(nr):
            euclid_heatmap[r1, r2] = np.linalg.norm(representations[r1, :] - representations[r2, :])

    simple_plot(euclid_heatmap, mode="reds")


def plot_pr_curve(precision, recall):
    """
    plot_pr_curve(precision, recall) -> No return object
    Plot PR curve.
    Args:
        precision (List): List of precision values
        recall (List): List of recall values
    """

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.savefig('XGBoost_PR')
    plt.show()


def plot_confusion_matrix(predictions):
    """
    plot_confusion_matrix(predictions) -> No return object
    Plot confusion matrix for subcompartments.
    Args:
        predictions (DataFrame): frame of true and predicted subcompartments
    """

    conf_matrix = confusion_matrix(predictions[:, 7], predictions[:, 6])
    conf_matrix = conf_matrix[1:, 1:]
    df_cm = pd.DataFrame(conf_matrix)
    df_cm = df_cm.div(df_cm.sum(axis=0), axis=1)

    x_axis_labels = ["A2", "A1", "B1", "B2", "B3"]
    y_axis_labels = ["A2",
                     "A1", "B1", "B2", "B3"]

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", xticklabels=x_axis_labels,
               yticklabels=y_axis_labels)
    plt.show()


def plot_combined(map_frame):
    """
    plot_combined(map_frame) -> No return object
    Plot map for tasks
    Args:
        map_frame (DataFrame): dataframe of map values
    """
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


def plot_gbr(main_df):
    """
    plot_gbr(main_df) -> No return object
    Gets violin plots of Segway GBR
    Args:
        main_df (DataFrame): DF containing if values and targets
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
    """
    plot_r2(comb_r2_df) -> No return object
    plots average R2 values at a particular difference.
    Args:
        comb_r2_df (DataFrame): DF containing R2 values for various differences in positions.
    """

    max_diff = int(comb_r2_df['diff'].max())
    max_mb = 100
    num_bins_1mb = 100
    pos = np.arange(0, max_mb)
    avg_diff = pd.DataFrame(columns=["diff", "r2"])
    r2_list = []
    r2_list_pos = []

    "get average r2"
    for diff in range(max_diff):
        subset_diff = comb_r2_df.loc[comb_r2_df["diff"] == diff]
        r2_mean = subset_diff["r2"].mean()
        avg_diff = avg_diff.append({"diff": diff, "r2": r2_mean}, ignore_index=True)

    "mean in window"
    for i in range(int(np.ceil(max_diff/num_bins_1mb))):
        r2_sub = avg_diff.loc[(avg_diff["diff"] >= i*num_bins_1mb) & (avg_diff["diff"] < (i+1)*num_bins_1mb)]
        r2_mean = r2_sub["r2"].mean(skipna=True)
        r2_list.append(r2_mean)

    num_windows = int(np.ceil(len(r2_list)/max_mb))
    if num_windows == 1:
        r2_list_pos = np.zeros((num_windows, len(r2_list)))
    else:
        r2_list_pos = np.zeros((num_windows, max_mb))
    for k in range(num_windows):
        if k == num_windows - 1:
            r2_list_pos[k] = r2_list[k * max_mb: ]
        else:
            r2_list_pos[k] = r2_list[k * max_mb: (k + 1) * max_mb]

    r2_list_pos = np.mean(r2_list_pos, axis=0)

    "plot"
    plt.figure(figsize=(12, 10))
    plt.plot(pos, r2_list_pos, marker='', markersize=14, color='C0', label='Hi-C-LSTM')
    plt.tick_params(axis="x", labelsize=20, length=0)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel('Distance between positions in Mbp', fontsize=20)
    plt.ylabel('R-squared for Replicate-1', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()


def scatter_tal_lm(ko, wt):
    """
    scatter_tal_lm(ko, wt) -> No return object
    Scatter plot of TAL1 and LMO2 prediction differences.
    Args:
        ko (Array): Array containing after knockout values
        wt (Array): Array containing before knockout values
    """

    def func(x, a):
        return a * x

    diff_mat = ko - wt
    diff_mat[0,0] = 0
    og = np.triu(diff_mat)
    og = og.flatten(order='C')
    pred = np.triu(diff_mat.T)
    pred = pred.flatten(order='C')

    plt.figure(figsize=(10, 8))
    #res = sm.OLS(pred, og).fit()
    m, _ = curve_fit(func, og, pred)
    plt.scatter(og, pred, marker='o', alpha=0.5)
    plt.plot(og, m*og, "g")
    # sns.regplot(og, pred)
    plt.tick_params(axis="x", labelsize=20, length=0)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    plt.xlabel('TAL1 KO - WT (Observed)', fontsize=20)
    plt.ylabel('TAL1 KO - WT (Predicted)', fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/kevindsouza/Downloads/tal1_scatter.png")


def hist_2d(og, pred):
    """
    hist_2d(og, pred) -> No return object
    2D histogram of observed and predicted differences.
    Args:
        og (Array): Array containing observed differences
        pred (Array): Array containing predicted differences
    """
    x_min = np.min(og)
    x_max = np.max(og)

    y_min = np.min(pred)
    y_max = np.max(pred)

    x_bins = np.linspace(x_min, x_max, 50)
    y_bins = np.linspace(y_min, y_max, 50)

    plt.figure(figsize=(10, 8))
    hist, _, _, _ = plt.hist2d(og, pred, bins=[x_bins, y_bins])
    plt.xticks(fontsize=18)
    plt.xlim([0, 0.1])
    plt.yticks(fontsize=18)
    plt.ylim([0.004, 0.1])
    plt.xlabel('LMO2 KO - WT (Original)', fontsize=20)
    plt.ylabel('LMO2 KO - WT (Predicted)', fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/kevindsouza/Downloads/lmo2_hist.png")


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell

    for chr in cfg.chr_test_list:
        '''
        r2_diff = pd.read_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")
        r2_diff = r2_diff.drop(['Unnamed: 0'], axis=1)
        comb_r2_df = comb_r2_df.append(r2_diff, ignore_index=True)
        plot_r2(comb_r2_df)
        '''

        pred_data = pd.read_csv(cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        hic_mat, st = get_heatmaps(pred_data, no_pred=False)
        simple_plot(hic_mat, mode="reds")

        '''
        foxg1_data = pd.read_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (cell, str(chr)), sep="\t")
        plot_foxg1(foxg1_data)
        '''

        '''
        shift_pad = np.load(cfg.output_directory + "ctcf_diff_shift_padding.npy")
        simple_plot(shift_pad, mode="diff")
        '''
        print("done")