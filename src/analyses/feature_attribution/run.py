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
from training.data_utils import get_cumpos
from analyses.plot.plot_utils import plot_gbr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_captum(cfg, model, chr):
    """
    run_captum(cfg, model, chr) -> DataFrame
    Gets data for chromosome and cell type. Runs IG using captum.
    Saves resulting IG DataFrame.
    Args:
        cfg (Config): The configuration to use for the experiment.
        model (SeqLSTM): The model to run captum on.
        chr (int): The chromosome to run captum on.
    """

    torch.manual_seed(123)
    np.random.seed(123)

    "get DataLoader"
    data_loader = get_data_loader_chr(cfg, chr, shuffle=False)

    "run IG"
    ig_df = model.get_captum_ig(data_loader)

    "save IG dataframe"
    np.save(cfg.output_directory + "ig_df_chr%s.npy" % (str(chr)), ig_df)
    return ig_df


def get_top_tfs_chip(cfg, ig_df, chr):
    """
    get_top_tfs_chip(cfg, ig_df, chr) -> DataFrame
    Attributed importance to each of the TFs according to CHipSeq Peaks.
    Args:
        cfg (Config): The configuration to use for the experiment.
        ig_df (DataFrame): Dataframe containing positions and IG values.
        chr (int): The chromosome to which IG values belong.
    """

    cumpos = get_cumpos(cfg, chr)

    tf_ob = TFChip(cfg, chr)
    chip_data = tf_ob.get_chip_data()

    chip_data = chip_data.drop_duplicates(keep='first').reset_index(drop=True)

    pos_chip_data = pd.DataFrame(columns=["pos", "target", "start", "end", "chr"])
    if chip_data.index[0] == 1:
        chip_data.index -= 1

    for i in range(0, chip_data.shape[0]):

        start = chip_data.loc[i, "start"]
        end = chip_data.loc[i, "end"]

        for j in range(start, end + 1):
            pos_chip_data = pos_chip_data.append(
                {'pos': j, 'target': chip_data.loc[i, "target"], 'start': chip_data.loc[i, "start_full"],
                 'end': chip_data.loc[i, "end_full"], 'chr': chip_data.loc[i, "chr"]}, ignore_index=True)

    pos_chip_data["pos"] = pos_chip_data["pos"] + cumpos
    ig_df = pd.merge(ig_df, pos_chip_data, on="pos")
    ig_df["pos"] = ig_df["pos"] - cumpos
    ig_df.reset_index(drop=True, inplace=True)
    return ig_df


def get_top_tfs_db(cfg, ig_df, chr):
    """
    get_top_tfs_db(cfg, ig_df, chr) -> DataFrame
    Attributed importance to each of the TFs according TF database from TFBS web portal.
    Args:
        cfg (Config): The configuration to use for the experiment.
        ig_df (DataFrame): Dataframe containing positions and IG values.
        chr (int): The chromosome to which IG values belong.
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
    cumpos = get_cumpos(cfg, chr)
    chr_start = [cumpos for _ in range(len(tf_db_chr))]
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
        chr (int): The chromosome to which IG values belong.
        ig_df (Dataframe): Dataframe containing positions and IG values
        element (string): one of Segway, GBR, CTCF, FIREs, TADs, Loop_Domains,
                        Domains, RAD21, SMC3, Merge_Domains, TADBs, TADBsCTCF+,
                        TADBsCTCF-, Loop_CTCFCohesin, NonLoop_CTCFCohesin.
    """

    "use downstream obejct to access helper"
    downstream_ob = DownstreamTasks(cfg)
    cumpos = get_cumpos(cfg, chr)

    ig_df = ig_df.astype({"pos": int})

    if element == "Segway":
        seg_ob = SegWay(cfg, chr)
        element_data = seg_ob.segway_small_annotations()

    elif element == "GBR":
        seg_ob = SegWay(cfg, chr)
        element_data = seg_ob.segway_gbr().reset_index(drop=True)

    elif element == "CTCF":
        ctcf_ob = TFChip(cfg, chr)
        element_data = ctcf_ob.get_ctcf_data()

    elif element == "FIREs":
        fire_ob = Fires(cfg, chr, mode="ig")
        fire_ob.get_fire_data()
        element_data = fire_ob.filter_fire_data()

    elif element == "TADs":
        domain_ob = Domains(cfg, chr, mode="ig")
        element_data = domain_ob.get_tad_data()

    elif element == "Loop_Domains":
        loop_ob = Loops(cfg, chr, mode="ig")
        element_data = loop_ob.get_loop_data()

    elif element == "Domains":
        domain_ob = Domains(cfg, chr, mode="ig")
        element_data = domain_ob.get_domain_data()

    elif element == "RAD21":
        tf_ob = TFChip(cfg, chr)
        element_data, _ = tf_ob.get_cohesin_data()

    elif element == "SMC3":
        tf_ob = TFChip(cfg, chr)
        _, element_data = tf_ob.get_cohesin_data()

    elif element == "Merge_Domains":
        domain_ob = Domains(cfg, chr, mode="ig")
        element_data = domain_ob.merge_domains()

    elif element == "TADBs":
        domain_ob = Domains(cfg, chr, mode="ig")
        tf_ob = TFChip(cfg, chr)
        element_data = domain_ob.get_tad_boundaries(tf_ob, ctcf="all")
        pass

    elif element == "TADBsCTCF+":
        domain_ob = Domains(cfg, chr, mode="ig")
        tf_ob = TFChip(cfg, chr)
        element_data = domain_ob.get_tad_boundaries(tf_ob, ctcf="positive")

    elif element == "TADBsCTCF-":
        domain_ob = Domains(cfg, chr, mode="ig")
        tf_ob = TFChip(cfg, chr)
        element_data = domain_ob.get_tad_boundaries(tf_ob, ctcf="negative")

    elif element == "Loop_CTCFCohesin":
        tf_ob = TFChip(cfg, chr)
        element_data = tf_ob.ctcf_in_loops(loops="inside")

    elif element == "NonLoop_CTCFCohesin":
        tf_ob = TFChip(cfg, chr)
        element_data = tf_ob.ctcf_in_loops(loops="outside")

    element_data = element_data.drop_duplicates(keep='first').reset_index(drop=True)
    if element != "TADBs" and element != "TADBsCTCF+" and element != "TADBsCTCF-":
        element_data = downstream_ob.downstream_helper_ob.get_window_data(element_data)
    element_data["pos"] = element_data["pos"] + cumpos
    ig_df = pd.merge(ig_df, element_data, on="pos")
    ig_df.reset_index(drop=True, inplace=True)

    return ig_df


def run_experiment(cfg, model):
    """
    run_experiment(cfg, model) -> No return object
    Runs experiment with specified config for all specified test chromosomes.
    If run_captum is True, runs captum on specifief chromosome. If False loads from saved IG DataFrame.
    If run_tfs is True, compares with TFBS positions.
    if run_elements is True, compares with psoitions of specified element.
    Use run_all_elements to run for all elements in elements list.
    Args:
        cfg (Config): The configuration to use for the experiment.
        model (SeqLSTM): Model to be used to run integrated gradients.
    """
    main_df = pd.DataFrame()
    for chr in cfg.decoder_test_list:
        print('IG Start Chromosome: {}'.format(chr))

        if cfg.run_captum:
            "run captum and save IG Dataframe"
            ig_df = run_captum(cfg, model, chr)
        else:
            "load saved IG dataframe"
            try:
                ig_df = pd.DataFrame(np.load(cfg.output_directory + "ig_df_chr%s.npy" % (str(chr))),
                                     columns=["ig", "pos"])
            except Exception as e:
                print("Make sure IG values are computed and saved")
                quit()

        "attribute TFs"
        if cfg.ig_run_tfs:
            if cfg.ig_run_chip:
                ig_elements = get_top_tfs_chip(cfg, ig_df, chr)
            else:
                ig_elements = get_top_tfs_db(cfg, ig_df, chr)
            main_df = pd.concat([main_df, ig_elements], axis=0)
        elif cfg.ig_run_elements:
            "attribute elements"
            ig_elements = attribute_elements(cfg, chr, ig_df, element=cfg.ig_element)
            main_df = pd.concat([main_df, ig_elements], axis=0)

    "sort TFs by IG values"
    if cfg.ig_run_tfs:
        if not cfg.ig_run_chip:
            main_df = main_df.groupby('target').agg({'ig': 'mean'})
        main_df = main_df.sort_values("ig", ascending=False)
        main_df.to_csv(cfg.output_directory + "ig_tf.csv", sep="\t")
    elif cfg.ig_run_elements:
        "save element IG"
        main_df.to_csv(cfg.output_directory + "ig_%s.csv" % cfg.ig_element, sep="\t")
    return main_df


def run_all_elements(cfg, model):
    """
    run_all_elements(cfg, model) -> No return object
    Runs experiment with specified config and chosen element. Run experiment runs for all specified test chromosomes.
    Args:
        cfg (Config): The configuration to use for the experiment.
        model (SeqLSTM): Model to be used to run integrated gradients.
    """

    cfg.run_elements = True
    cfg.run_tfs = False
    main_df_tasks = pd.DataFrame()
    for element in cfg.ig_elements_list:
        cfg.ig_element = element
        main_df = run_experiment(cfg, model)
        main_df_tasks = pd.concat([main_df_tasks, main_df], axis=0)

    main_df_tasks = model.compute_rowwise_ig(main_df_tasks)

    "plot"
    plot_gbr(main_df_tasks)


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "load model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    if cfg.ig_run_all_elements:
        "run all elements"
        run_all_elements(cfg, model)
    else:
        "run according to config"
        run_experiment(cfg, model)
