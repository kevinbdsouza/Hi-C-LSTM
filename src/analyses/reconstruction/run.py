import pandas as pd
from training.config import Config
import torch
from training.model import SeqLSTM
from analyses.reconstruction.hic_r2 import HiC_R2
from analyses.plot import plot_utils
from training.test_model import test_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    """
    Scipt to run test on an existing model and compute R2 along the genomic axis. 
    To use a different cell type, change cell type in Config. 
    Supports different cell types like GM12878, WTC11, HFF-hTERT, H1-hESC, and replicates.  
    Before you run test and compute R2, make sure the trained model exists for the cell type and given chromosome.
    To get data for various cell types, refer to the manuscript for links. Run juicer to extract the Hi-C datasets into txt files. 
    Use bash script extract_chromosomes.sh to extract hic file into txt files for specified chromosomes.
    
    To compute R2 with representations from SNIPER and SCA, obtain representations from SNIPER and SCA. 
    
    
    Once you compute R2 values for distances, you can use plot_r2 function in analyses/plot/plot_fns.py
    """

    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "initalize model"
    model = SeqLSTM(cfg, device, model_name).to(device)

    "load model weights"
    model.load_weights()

    comb_r2_df = pd.DataFrame(columns=["diff", "r2"])
    for chr in cfg.chr_test_lsit:
        "Run test and save the predictions"
        # test_model(model, cfg, cell, chr)

        "load predictions"
        hic_predictions = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)),
                                      sep="\t")
        hic_predictions = hic_predictions.drop(['Unnamed: 0'], axis=1)

        "compute R2 and save R2"
        hic_r2_ob = HiC_R2(cfg, chr, mode='test')
        r2_frame = hic_r2_ob.hic_r2(hic_predictions)
        r2_frame.to_csv(cfg.output_directory + "r2frame_%s_chr%s.csv" % (cell, str(chr)), sep="\t")
        comb_r2_df = comb_r2_df.append(r2_frame, ignore_index=True)

    "plot r2"
    plot_utils.plot_r2(comb_r2_df)

