from __future__ import division
import torch
import numpy as np
from training.config import Config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr
from analyses.feature_attribution.tf import TFChip
import pandas as pd
from analyses.feature_attribution.segway import SegWay
from analyses.classification.run import DownstreamTasks
from analyses.classification.fires import Fires
from analyses.classification.loops import Loops
from analyses.classification.domains import Domains
import seaborn as sns
import matplotlib.pyplot as plt
from training.data_utils import get_cumpos

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_captum(cfg, model, chr):
    """
    captum_test(cfg, model, chr) -> DataFrame
    Gets data for chromosome and cell type. Runs IG using captum.
    Args:
        cfg (Config): The configuration to use for the experiment.
        model (SeqLSTM): The model to run captum on.
        chr (int): The chromosome to run captum on.
    """

    torch.manual_seed(123)
    np.random.seed(123)

    "get DataLoader"
    data_loader = get_data_loader_chr(cfg, chr)

    "run IG"
    ig_df = model.get_captum_ig(data_loader)

    "save IG dataframe"
    np.save(cfg.output_directory + "ig_df_chr%s.npy" % (str(chr)), ig_df)
    return ig_df


def attribute_tfs(cfg, ig_df, chr):
    """
    attribute_tfs(cfg, ig_df, chr) -> DataFrame
    Attributed importance to each of the TFs.
    Args:
        cfg (Config): The configuration to use for the experiment.
        ig_df (DataFrame): Dataframe containing positions and IG values.
        chr (int): The chromosome to run captum on.
    """

    ig_df = ig_df.astype({"pos": int})

    "read TF file"
    tf_db = pd.read_csv(cfg.tf_file_path)
    tf_db = tf_db.filter(cfg.tf_columns, axis=1)

    "remove chr X and Y"
    tf_db = tf_db.loc[(tf_db['chromosome'] != 'X') & (tf_db['chromosome'] != 'Y')]

    "filter chr"
    tf_db = tf_db.astype({"chromosome": int, "start": int})
    tf_db_chr = tf_db.loc[tf_db["chromosome"] == chr]

    "convert TF positions to cumulative indices"
    chr_start = get_cumpos(cfg, chr)
    tf_db_chr["start"] = tf_db_chr["start"] + chr_start
    tf_db_chr = tf_db_chr.rename(columns={'start': 'pos', 'HGNC symbol': 'target'})

    "merge DFs on position"
    comb_df = pd.merge(tf_db_chr, ig_df, on="pos")
    return comb_df


def attribute_elements(cfg, chr, ig_df, element="ctcf"):
    """
    attribute_elements(cfg, chr, ig_df, element) -> DataFrame
    Merges data for specified element with IG values based on position.
    Args:
        cfg (Config): The configuration to use for the experiment.
        chr (int): The chromosome to run captum on.
        ig_df (Dataframe): Dataframe containing positions and IG values
        element (string): one of small, gbr, ctcf, fire, tad, loops, domains, cohesin
    """

    "use downstream obejct to access helper"
    downstream_ob = DownstreamTasks(cfg, chr, mode='lstm')
    cumpos = get_cumpos(cfg, chr)

    ig_df = ig_df.astype({"pos": int})

    if element == "small":
        seg_ob = SegWay(cfg, chr)
        element_data = seg_ob.segway_small_annotations()

    elif element == "gbr":
        seg_ob = SegWay(cfg, chr)
        element_data = seg_ob.segway_gbr().reset_index(drop=True)

    elif element == "ctcf":
        ctcf_ob = TFChip(cfg, chr)
        element_data = ctcf_ob.get_ctcf_data()

    elif element == "fire":
        fire_ob = Fires(cfg, chr)
        fire_ob.get_fire_data()
        element_data = fire_ob.filter_fire_data()

    elif element == "tad":
        fire_ob = Fires(cfg, chr)
        element_data = fire_ob.get_tad_data()

    elif element == "loops":
        loop_ob = Loops(cfg, chr, mode="ig")
        element_data = loop_ob.get_loop_data()

    elif element == "domains":
        domain_ob = Domains(cfg, chr, mode="ig")
        element_data = domain_ob.get_domain_data()

    elif element == "rad21":
        tf_ob = TFChip(cfg, chr)
        element_data, _ = tf_ob.get_cohesin_data()

    elif element == "smc3":
        tf_ob = TFChip(cfg, chr)
        _, element_data = tf_ob.get_cohesin_data()

    element_data = element_data.drop_duplicates(keep='first').reset_index(drop=True)
    element_data = downstream_ob.downstream_helper_ob.get_window_data(element_data)
    element_data["pos"] = element_data["pos"] + cumpos
    ig_df = pd.merge(ig_df, element_data, on="pos")
    ig_df.reset_index(drop=True, inplace=True)

    return ig_df


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
    pass


if __name__ == '__main__':

    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "load model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    main_df = pd.DataFrame(columns=["pos", "target"])
    for chr in cfg.decoder_test_list:
        print('IG Start Chromosome: {}'.format(chr))

        if cfg.run_captum:
            "run captum and save IG Dataframe"
            ig_df = run_captum(cfg, model, chr)
        else:
            "load saved IG dataframe"
            try:
                ig_df = pd.DataFrame(np.load(cfg.output_directory + "ig_df_chr%s.npy" % (str(chr))),
                                     columns=["pos", "ig"])
            except Exception as e:
                print("Make sure IG values are computed and saved")

        "attribute TFs"
        if cfg.run_tfs:
            ig_elements = attribute_tfs(cfg, ig_df, chr)
            ig_elements.to_csv(cfg.output_directory + "ig_tf%s.csv" % (str(chr)), sep="\t")

        "attribute elements"
        if cfg.run_elements:
            ig_elements = attribute_elements(cfg, chr, ig_df, element=cfg.element)
            ig_elements.to_csv(cfg.output_directory + "ig_%s_chr%s.csv" % (cfg.element, str(chr)), sep="\t")

        main_df = pd.concat([main_df, ig_elements], axis=0)

    main_df = main_df.groupby('target').agg({'ig': 'mean'})
    main_df = main_df.sort_values("ig")
    main_df = main_df.iloc[-5:][ :]
    plot_gbr(main_df)
