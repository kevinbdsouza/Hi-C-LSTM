import torch
import training.config as config
import numpy as np
import pandas as pd
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from training.data_utils import load_hic, contactProbabilities, get_bin_idx
from analyses.plot.plot_utils import get_heatmaps
from sklearn.decomposition import PCA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_pca(cfg, chr):
    """
    pca(cfg, chr) -> DataFrame
    Method to compute and save pca of Hi-C data.
    Return
    Args:
        cfg (Config): configuration to use for PCA.
        chr (int): chromsome to run PCA on.
    """

    pca_columns = list(np.arange(cfg.pos_embed_size)) + ["i"]
    pred_df = pd.DataFrame(columns=pca_columns)

    data = load_hic(cfg, chr)
    data['i_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['i'], cfg)
    data['j_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['j'], cfg)
    data["v"] = contactProbabilities(data["v"])
    hic_mat, st = get_heatmaps(data)

    pca_ob = PCA(n_components=cfg.pos_embed_size)
    pca_ob.fit(hic_mat)
    hic_mat = pca_ob.transform(hic_mat)

    return pred_df


def test_model(model, cfg, chr):
    """
    train_model(model, cfg, cell, chr) -> No return object
    Loads loads data, tests the model, and saves the predictions in a csv file.
    Works on one chromosome at a time.
    Args:
        model (SeqLSTM): The model that needs to be tested.
        cfg (Config): The configuration to use for the experiment.
        cell (string): The cell type to extract Hi-C from.
        chr (int): The chromosome to test.
    """

    if not cfg.compute_pca:
        "get data"
        data_loader = get_data_loader_chr(cfg, chr)

        "test model"
        predictions, test_error, values, pred_df, error_list = model.test(data_loader)

        "save predictions"
        pred_df.to_csv(cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")
    else:
        pred_df = compute_pca(cfg, chr)


if __name__ == '__main__':
    cfg = config.Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "Initalize Model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    for chr in cfg.chr_test_list:
        test_model(model, cfg, chr)
