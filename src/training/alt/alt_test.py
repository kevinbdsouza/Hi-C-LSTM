import torch
from training.alt.alt_config import Config
import numpy as np
import pandas as pd
from training.alt.alt_model import SeqLSTM
from training.alt.alt_data_utils import get_data_loader_chr
from training.data_utils import load_hic, contactProbabilities, get_bin_idx
from analyses.plot.plot_utils import get_heatmaps, simple_plot
from sklearn.decomposition import PCA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    if cfg.full_test:
        "test model"
        og_mat, pred_mat = model.test()

        simple_plot(pred_mat)
        simple_plot(og_mat)

        print("done")
        "save predictions"
        #pred_df.to_csv(cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")
    elif cfg.get_zero_pred:
        "zero pred"
        data_loader = get_data_loader_chr(cfg, chr, shuffle=False)
        zero_embed = model.zero_embed(data_loader)
        return zero_embed


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "Initalize Model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    for chr in cfg.chr_test_list:
        test_model(model, cfg, chr)
