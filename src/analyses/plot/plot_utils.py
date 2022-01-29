import pandas as pd
import numpy as np
import seaborn as sns
from training.data_utils import contactProbabilities
#import matplotlib as mpl

#mpl.use('module://backend_interagg')
import training.config as config
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from analyses.classification.domains import Domains


def plot_heatmaps(data):
    st = int(data["i"].min())
    data["i"] = data["i"] - st
    data["j"] = data["j"] - st
    nr = int(data["j"].max()) + 1
    rows = np.array(data["i"]).astype(int)
    cols = np.array(data["j"]).astype(int)

    hic_mat = np.zeros((nr, nr))
    hic_mat[rows, cols] = np.array(contactProbabilities(data["v"]))
    hic_upper = np.triu(hic_mat)
    hic_mat[cols, rows] = np.array(data["pred"])
    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2
    # hic_win = hic_mat[6701:7440, 6701:7440]
    # hic_win = hic_mat[900:1450, 900:1450]

    simple_plot(hic_mat)
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
    hic_mat[cols, rows] = np.array(data["ko_pred"])
    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2

    simple_plot(hic_mat)
    pass


def plot_tal1_lmo2(data):
    tal_data = data.loc[data["i"] < 5000]
    lmo2_data = data.loc[data["i"] > 5000]

    plot_heatmaps(tal_data)
    # plot_heatmaps(lmo2_data)
    pass


def simple_plot(hic_win):
    '''
    plt.imshow(hic_win, cmap='hot', interpolation='nearest')
    plt.yticks([])
    plt.xticks([])
    plt.show()
    '''

    sns.set_theme()
    ax = sns.heatmap(hic_win, cmap="Reds", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
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


def plot_r2(comb_r2_df):
    max_diff = 9008
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
        r2_list_pos.append(np.mean(r2_list[k * 5: (k + 1) * 5]))

    plt.figure(figsize=(12, 10))
    plt.plot(pos, r2_list_pos, marker='', markersize=14, color='C0', label='Hi-C-LSTM')
    plt.tick_params(axis="x", labelsize=20, length=0)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel('Distance between positions in Mbp', fontsize=20)
    plt.ylabel('R-squared for Replicate-1', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.show()

    pass


if __name__ == '__main__':
    plot_chr = list(range(1, 2))
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

        '''
        pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
        plot_foxg1(pred_data)
        foxg1_data = pd.read_csv(cfg.output_directory + "shuffle_%s_afko_chr%s.csv" % (cell, str(chr)), sep="\t")
        plot_foxg1(foxg1_data)
        '''

        #foxg1_ko = np.load(cfg.output_directory + "foxg1_ko.npy")
        #simple_plot(foxg1_ko)

        #foxg1_ko = np.load(cfg.output_directory + "tal1_wt.npy")
        #simple_plot(foxg1_ko)
        #pred_data = pd.read_csv(cfg.output_directory + "%s_predictions_chr.csv" % (cell), sep="\t")
        #plot_tal1_lmo2(pred_data)

        tal_ko = pd.read_csv(cfg.hic_path + cell +"/talko_tal_df.txt", sep="\t")
        plot_heatmaps(tal_ko)

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
