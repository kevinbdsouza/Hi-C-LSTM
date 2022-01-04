import pandas as pd
from training.config import Config
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

    comb_r2_df = pd.DataFrame(columns=["diff", "r2"])
    for chr in test_chr:
        # Run test and save the predictions
        # test_model(model, cfg, cell, chr)

        # load predictions, compute r2 and save
        hic_r2_ob = HiC_R2(cfg, chr, mode='test')
        hic_predictions = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)),
                                      sep="\t")
        hic_predictions = hic_predictions.drop(['Unnamed: 0'], axis=1)
        r2_frame = hic_r2_ob.hic_r2(hic_predictions)
        r2_frame.to_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")
        comb_r2_df = comb_r2_df.append(r2_frame, ignore_index=True)

    # plot r2
    plot_utils.plot_r2(comb_r2_df)

    # Other R2 plots can be obtained by:
    # Changing the cell type,
    # The model associated with the cell type,
    # Ablations of the model,
    # Other models like Sniper and SCA,
    # Changing the data to the replicate

    # If you have all the data, you can use plot_r2 function in analyses/plot/plot_fns.py
