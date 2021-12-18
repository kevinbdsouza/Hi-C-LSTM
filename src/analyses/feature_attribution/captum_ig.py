from __future__ import division
import traceback
import torch
import numpy as np
import os
import pandas as pd
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from training.data_utils import get_bin_idx

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# device = 'cpu'
mode = "test"


def captum_test(cfg, model, cell, chr):
    torch.manual_seed(123)
    np.random.seed(123)

    data_loader, samples = get_data_loader_chr(cfg, cell, chr)
    ig_df = model.get_captum_ig(data_loader)

    np.save(cfg.output_directory + "ig_df_chr" + str(chr) + ".npy", ig_df)
    return ig_df


def captum_analyze(cfg, model, cell, chr):
    prediction_path = "/data2/hic_lstm/downstream/predictions/"
    columns = ["HGNC symbol", "chromosome", "start"]
    ig_df = pd.DataFrame(np.load(prediction_path + "ig_df_chr" + str(chr) + ".npy"),
                         columns=["start", "ig"])
    ig_df = ig_df.astype({"start": int})

    tf_db = pd.read_csv(prediction_path + "/tf_db.csv")
    tf_db = tf_db.filter(columns, axis=1)
    tf_db = tf_db.loc[(tf_db['chromosome'] != 'X') & (tf_db['chromosome'] != 'Y')]
    tf_db = tf_db.astype({"chromosome": int, "start": int})
    tf_db_chr = tf_db.loc[tf_db["chromosome"] == chr]

    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr_start = sizes["chr"+str(chr-1)]
    tf_db_chr["start"] = tf_db_chr["start"] + chr_start

    comb_df = pd.merge(tf_db_chr, ig_df, on="start")
    pass


if __name__ == '__main__':

    cfg = config.Config()
    cell = "GM12878"

    # load model
    model_name = "shuffle2_og"
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    #test_chr = list(range(5, 11))
    # test_chr.remove(11)
    test_chr = [22]

    lstm_hidden_states = np.zeros((cfg.genome_len, cfg.hidden_size_lstm))
    # embed_rows = np.load(cfg.output_directory + "embeddings.npy")

    for chr in test_chr:
        print('Testing Start Chromosome: {}'.format(chr))

        #feature_importance = captum_test(cfg, model, cell, chr)

        captum_analyze(cfg, model, cell, chr)

    print("done")
