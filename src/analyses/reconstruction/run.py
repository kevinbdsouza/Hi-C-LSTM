import pandas as pd
from training.config import Config
from training.test_model import test_model
import torch
from training.model import SeqLSTM
from analyses.reconstruction.hic_r2 import HiC_R2
from analyses.plot import plot_utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    test_chr = list(range(22, 23))
    cfg = Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    # initalize model
    model = SeqLSTM(cfg, device, model_name).to(device)

    # load model weights
    model.load_weights()

    for chr in test_chr:
        test_model(model, cfg, cell, chr)
        data_ob_hic = HiC_R2(cfg, chr, mode='test')