from __future__ import division
import torch
import numpy as np
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from analyses.feature_attribution.tf import TFChip
import pandas as pd
from analyses.feature_attribution.segway import SegWay
from training.config import Config
from analyses.classification.run import DownstreamTasks
from analyses.classification.fires import Fires
from analyses.classification.loops import Loops
from analyses.classification.domains import Domains

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
mode = "test"


def captum_test(cfg, model, cell, chr):
    torch.manual_seed(123)
    np.random.seed(123)

    data_loader, samples = get_data_loader_chr(cfg, chr)
    ig_df = model.get_captum_ig(data_loader)

    np.save(cfg.output_directory + "ig_df_chr" + str(chr) + ".npy", ig_df)
    return ig_df


def captum_analyze_tfs(cfg, ig_df, chr):
    columns = ["HGNC symbol", "chromosome", "start"]
    prediction_path = "/data2/hic_lstm/downstream/predictions/"
    ig_df = ig_df.astype({"pos": int})

    tf_db = pd.read_csv(prediction_path + "/tf_db.csv")
    tf_db = tf_db.filter(columns, axis=1)
    tf_db = tf_db.loc[(tf_db['chromosome'] != 'X') & (tf_db['chromosome'] != 'Y')]
    tf_db = tf_db.astype({"chromosome": int, "start": int})
    tf_db_chr = tf_db.loc[tf_db["chromosome"] == chr]

    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr_start = sizes["chr" + str(chr - 1)]
    tf_db_chr["start"] = tf_db_chr["start"] + chr_start
    tf_db_chr = tf_db_chr.rename(columns={'start': 'pos'})

    comb_df = pd.merge(tf_db_chr, ig_df, on="pos")
    return comb_df


def captum_analyze_elements(cfg, chr, ig_df, mode):
    downstream_ob = DownstreamTasks(cfg, chr, mode='lstm')
    ig_df = ig_df.astype({"pos": int})

    chr_seg = 'chr' + str(chr)
    segway_cell_names = ["GM12878"]
    chr_ctcf = 'chr' + str(chr)
    chr_fire = chr
    chr_tad = 'chr' + str(chr)
    fire_cell_names = ['GM12878']
    fire_path = cfg.downstream_dir + "FIREs"

    if mode == "small":
        seg_ob = SegWay(cfg, chr_seg, segway_cell_names)
        annotations = seg_ob.segway_small_annotations()
        annotations = downstream_ob.downstream_helper_ob.get_window_data(annotations)
        annotations["pos"] = annotations["pos"] + downstream_ob.downstream_helper_ob.start
        ig_df = pd.merge(ig_df, annotations, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "gbr":
        seg_ob = SegWay(cfg, chr_seg, segway_cell_names)
        annotations = seg_ob.segway_gbr().reset_index(drop=True)
        annotations = downstream_ob.downstream_helper_ob.get_window_data(annotations)
        annotations["pos"] = annotations["pos"] + downstream_ob.downstream_helper_ob.start
        ig_df = pd.merge(ig_df, annotations, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "ctcf":
        cell = "GM12878"
        ctcf_ob = TFChip(cfg, cell, chr_ctcf)
        ctcf_data = ctcf_ob.get_ctcf_data()
        ctcf_data = ctcf_data.drop_duplicates(keep='first').reset_index(drop=True)
        ctcf_data = downstream_ob.downstream_helper_ob.get_window_data(ctcf_data)
        ctcf_data["pos"] = ctcf_data["pos"] + downstream_ob.downstream_helper_ob.start
        ig_df = pd.merge(ig_df, ctcf_data, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "fire":
        fire_ob = Fires(cfg)
        fire_ob.get_fire_data(fire_path)
        fire_labeled = fire_ob.filter_fire_data(chr_fire)
        fire_window_labels = fire_labeled.filter(['start', 'end', "GM12878" + '_l'], axis=1)
        fire_window_labels.rename(columns={"GM12878" + '_l': 'target'}, inplace=True)
        fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

        fire_window_labels = downstream_ob.downstream_helper_ob.get_window_data(fire_window_labels)
        fire_window_labels["pos"] = fire_window_labels["pos"] + downstream_ob.downstream_helper_ob.start
        ig_df = pd.merge(ig_df, fire_window_labels, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "tad":
        fire_ob = Fires(cfg)
        fire_ob.get_tad_data(fire_path, fire_cell_names)
        tad_cell = fire_ob.filter_tad_data(chr_tad)[0]
        tad_cell['target'] = 1
        tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
        tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)

        tad_cell = downstream_ob.downstream_helper_ob.get_window_data(tad_cell)
        tad_cell["pos"] = tad_cell["pos"] + downstream_ob.downstream_helper_ob.start

        ig_df = pd.merge(ig_df, tad_cell, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "loops":
        loop_ob = Loops(cfg, "GM12878", chr)
        loop_data = loop_ob.get_loop_data()

        pos_matrix = pd.DataFrame()
        for i in range(2):
            if i == 0:
                temp_data = loop_data.rename(columns={'x1': 'start', 'x2': 'end'},
                                             inplace=False)
            else:
                temp_data = loop_data.rename(columns={'y1': 'start', 'y2': 'end'},
                                             inplace=False)

            temp_data = temp_data.filter(['start', 'end', 'target'], axis=1)
            temp_data = temp_data.drop_duplicates(keep='first').reset_index(drop=True)

            temp_data = downstream_ob.downstream_helper_ob.get_window_data(temp_data)
            temp_data["pos"] = temp_data["pos"] + downstream_ob.downstream_helper_ob.start
            pos_matrix = pos_matrix.append(temp_data)

        ig_df = pd.merge(ig_df, pos_matrix, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "domains":
        domain_ob = Domains(cfg, "GM12878", chr)
        domain_data = domain_ob.get_domain_data()
        domain_data.rename(columns={'x1': 'start', 'x2': 'end'},
                           inplace=True)
        domain_data = domain_data.filter(['start', 'end', 'target'], axis=1)
        domain_data = domain_data.drop_duplicates(keep='first').reset_index(drop=True)

        domain_data = downstream_ob.downstream_helper_ob.get_window_data(domain_data)
        domain_data["pos"] = domain_data["pos"] + downstream_ob.downstream_helper_ob.start

        ig_df = pd.merge(ig_df, domain_data, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    elif mode == "cohesin":
        cell = "GM12878"
        cohesin_df = pd.DataFrame()
        tf_ob = TFChip(cfg, cell, chr_ctcf)
        rad_data, smc_data = tf_ob.get_cohesin_data()
        rad_data = rad_data.drop_duplicates(keep='first').reset_index(drop=True)
        smc_data = smc_data.drop_duplicates(keep='first').reset_index(drop=True)

        rad_data = downstream_ob.downstream_helper_ob.get_window_data(rad_data)
        rad_data["pos"] = rad_data["pos"] + downstream_ob.downstream_helper_ob.start
        rad_data["target"] = "RAD21"
        cohesin_df = cohesin_df.append(rad_data)

        smc_data = downstream_ob.downstream_helper_ob.get_window_data(smc_data)
        smc_data["pos"] = smc_data["pos"] + downstream_ob.downstream_helper_ob.start
        smc_data["target"] = "SMC3"
        cohesin_df = cohesin_df.append(smc_data)

        ig_df = pd.merge(ig_df, cohesin_df, on="pos")
        ig_df.reset_index(drop=True, inplace=True)

    return ig_df


if __name__ == '__main__':

    cfg = config.Config()

    # load model
    cell = cfg.cell
    model_name = "shuffle_" + cell
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    # test_chr = list(range(5, 11))
    test_chr = [22]

    for chr in test_chr:
        print('Testing Start Chromosome: {}'.format(chr))

        # ig_df = captum_test(cfg, model, cell, chr)
        prediction_path = "/data2/hic_lstm/downstream/predictions/"
        ig_df = pd.DataFrame(np.load(prediction_path + "ig_df_chr" + str(chr) + ".npy"),
                             columns=["pos", "ig"])

        ig_filtered_df = captum_analyze_tfs(cfg, ig_df, chr)
        # ig_filtered_df = captum_analyze_elements(cfg, chr, ig_df, mode="ctcf")

    print("done")
