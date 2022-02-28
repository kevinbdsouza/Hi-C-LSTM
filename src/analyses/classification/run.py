import logging
from analyses.classification.rna_seq import GeneExp
from training.config import Config
import pandas as pd
from analyses.classification.downstream_helper import DownstreamHelper
from analyses.classification.fires import Fires
from analyses.classification.rep_timing import Rep_timing
from analyses.classification.pe_interactions import PeInteractions
from analyses.classification.loops import Loops
from analyses.classification.domains import Domains
from analyses.classification.subcompartments import Subcompartments
from analyses.feature_attribution.tf import TFChip
from training.model import SeqLSTM
import torch
from training.test_model import test_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DownstreamTasks:
    """
    Class to run all downstream classification experiments using XGBoost.
    Compute mAP, Accuaracy, PR curves, AuROC, and F-score.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_model_dir = cfg.model_dir
        self.calculate_map = True
        self.downstream_helper_ob = DownstreamHelper(cfg)
        self.df_columns = [str(i) for i in range(0, 16)] + ["i"]

        self.subcompartment_path = cfg.downstream_dir + "subcompartments"
        self.chr_ctcf = 'chr' + str(chr)

    def run_xgboost(self, embed_rows, window_labels, chr, zero_target=True, mode="ends"):
        """
        run_xgboost(window_labels, chr) -> float, float, float, float
        Converts to cumulative indices. Depending on experiment, either runs baseline.
        Or runs classification using representations from chosen method and cell type.
        Return classifcation metrics.
        Args:
            window_labels (DataFrame): Contains start, end, target
            chr (int): chromosome to run xgboost on.
        """
        map, accuracy, f_score, auroc = 0, 0, 0, 0
        window_labels = window_labels.drop_duplicates(keep='first').reset_index(drop=True)

        if zero_target:
            col_list = ['start', 'end']
            zero_pos_frame = self.downstream_helper_ob.get_zero_pos(window_labels, col_list, chr)

        window_labels = self.downstream_helper_ob.add_cum_pos(window_labels, chr, mode=mode)

        if self.cfg.class_experiment == "subc_baseline":
            feature_matrix = self.downstream_helper_ob.subc_baseline(window_labels, mode="ends")
        elif self.cfg.class_experiment == "pca_baseline":
            feature_matrix = self.downstream_helper_ob.subc_baseline(window_labels, mode="ends")
        else:
            feature_matrix = self.downstream_helper_ob.get_feature_matrix(embed_rows, window_labels, chr)

        if zero_target:
            features = pd.DataFrame()
            features = features.append(feature_matrix)
            zero_features = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, chr, mode="pos")

            if self.cfg.class_experiment == "baseline":
                zero_features = self.downstream_helper_ob.subc_baseline(zero_features, mode="pos")
            else:
                zero_features = self.downstream_helper_ob.merge_features_target(embed_rows, zero_features)

            features = features.append(zero_features)
            feature_matrix = features.copy()

        if self.cfg.compute_metrics:
            try:
                self.downstream_helper_ob.cfg.class_mode = self.cfg.class_mode
                map, accuracy, f_score, auroc = self.downstream_helper_ob.calculate_map(feature_matrix)
            except Exception as e:
                print(e)
                return map, accuracy, f_score, auroc

        return map, accuracy, f_score, auroc

    def run_gene_expression(self, chr, embed_rows):
        """
        run_gene_expression(chr, embed_rows) -> float, float, float, float
        Gets gene expression data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("Gene Expression start")

        rna_seq_ob = GeneExp(self.cfg, chr)
        rna_seq_ob.get_rna_seq()
        rna_seq_chr = rna_seq_ob.filter_rna_seq()

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, rna_seq_chr, chr, zero_target=False, mode="ends")
        return map, accuracy, f_score, auroc

    def run_rep_timing(self, chr, embed_rows):
        """
        run_rep_timing(chr, embed_rows) -> float, float, float, float
        Gets replication timing data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("Replication Timing start")

        rep_ob = Rep_timing(self.cfg, chr)
        rep_chr = rep_ob.get_rep_data()

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, rep_chr, chr, zero_target=False, mode="ends")
        return map, accuracy, f_score, auroc

    def run_enhancers(self, chr, embed_rows):
        """
        run_enhancers(chr, embed_rows) -> float, float, float, float
        Gets Enhancer data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("Enhancer start")

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, pe_chr, chr, zero_target=True, mode="ends")
        return map, accuracy, f_score, auroc

    def run_tss(self, chr, embed_rows):
        """
        run_tss(chr, embed_rows) -> float, float, float, float
        Gets TSS data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("TSS start")

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, pe_chr, chr, zero_target=True, mode="ends")
        return map, accuracy, f_score, auroc

    def run_pe(self, chr, embed_rows):
        """
        run_pe(chr, embed_rows) -> float, float, float, float
        Gets PE interaction data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("PE start")

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, pe_chr, chr, zero_target=False, mode="ends")
        return map, accuracy, f_score, auroc

    def run_fires(self, chr, embed_rows):
        """
        run_fires(chr, embed_rows) -> float, float, float, float
        Gets FIRE data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("FIRE start")

        fire_ob = Fires(cfg, chr, mode="class")
        fire_ob.get_fire_data()
        fire_chr = fire_ob.filter_fire_data()

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, fire_chr, chr, zero_target=False, mode="ends")
        return map, accuracy, f_score, auroc

    def run_domains(self, chr, embed_rows):
        """
        run_domains(chr, embed_rows) -> float, float, float, float
        Gets TAD, subTAD, and boundaries data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("Domain start")

        domain_ob = Domains(cfg, chr, mode="class")
        self.cfg.class_mode = "binary"

        "get data and run xgboost according to type of domain element"
        if self.cfg.class_element == "TADs":
            domain_chr = domain_ob.get_tad_data()
            map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, domain_chr, chr, zero_target=True, mode="ends")
        elif self.cfg.class_element == "subTADs":
            domain_chr = domain_ob.get_tad_data()
            map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, domain_chr, chr, zero_target=True, mode="ends")
        elif self.cfg.class_element == "TADBs":
            tf_ob = TFChip(cfg, chr)
            tadb_chr = domain_ob.get_tad_boundaries(tf_ob, ctcf="all")
            map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, tadb_chr, chr, zero_target=True, mode="ends")
        elif self.cfg.class_element == "subTADBs":
            domain_chr = domain_ob.get_tad_data()
            map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, domain_chr, chr, zero_target=True, mode="ends")

        return map, accuracy, f_score, auroc

    def run_loops(self, chr, embed_rows):
        """
        run_loops(chr, embed_rows) -> float, float, float, float
        Gets loop data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classification metrics.
        Args:
            chr (int): chromosome to run classification on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        print("Loop Domain start")

        loop_ob = Loops(cfg, chr, mode="class")
        loop_chr = loop_ob.get_loop_data()

        "runs xgboost"
        self.cfg.class_mode = "binary"
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, loop_chr, chr, zero_target=True, mode="ends")
        return map, accuracy, f_score, auroc

    def run_sub_compartments(self, cfg):
        logging.info("subcompartment start")

        for cell in self.sbc_cell_names:
            sc_ob = Subcompartments(cfg, cell, chr, mode="Rao")
            sc_data = sc_ob.get_sc_data()

            sc_data = sc_data.filter(['start', 'end', 'target'], axis=1)
            sc_data = sc_data.drop_duplicates(keep='first').reset_index(drop=True)

            sc_data = self.downstream_helper_ob.add_cum_pos(sc_data, mode="ends")

            if self.exp == "baseline":
                feature_matrix = self.downstream_helper_ob.subc_baseline(Subcompartments, sc_data, mode="ends")
            else:
                feature_matrix = self.downstream_helper_ob.get_feature_matrix(sc_data)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="multi", exp=self.exp)

        return mean_map

    def run_experiment(self, model):
        """

        """
        cfg = self.cfg
        map_frame = pd.DataFrame(columns=["chr", "map", "fscore", "accuracy", "auroc"])
        for chr in cfg.decoder_test_list:
            print("Classification start Chromosome: {}".format(chr))

            if self.cfg.class_compute_representation:
                "running test model to get representations"
                test_model(model, cfg, chr)

            "load representations and filter"
            embed_rows = pd.read_csv(
                cfg.output_directory + "%s_%s_predictions_chr%s.csv" % (cfg.class_method, cfg.cell, str(chr)),
                sep="\t")
            embed_rows = embed_rows[self.df_columns]
            embed_rows = embed_rows.rename(columns={"i": "pos"})
            embed_rows = embed_rows.drop_duplicates(keep='first').reset_index(drop=True)

            "run element"
            if cfg.class_element == "Gene Expression":
                map, accuracy, f_score, auroc = self.run_gene_expression(chr, embed_rows)
            elif cfg.class_element == "Replication Timing":
                map, accuracy, f_score, auroc = self.run_rep_timing(chr, embed_rows)
            elif cfg.class_element == "Enhancers":
                map, accuracy, f_score, auroc = self.run_enhancers(chr, embed_rows)
            elif cfg.class_element == "TSS":
                map, accuracy, f_score, auroc = self.run_tss(chr, embed_rows)
            elif cfg.class_element == "PE-Interactions":
                map, accuracy, f_score, auroc = self.run_pe(chr, embed_rows)
            elif cfg.class_element == "FIREs":
                map, accuracy, f_score, auroc = self.run_fires(chr, embed_rows)
            elif cfg.class_element == "TADs":
                map, accuracy, f_score, auroc = self.run_domains(chr, embed_rows)
            elif cfg.class_element == "subTADs":
                map, accuracy, f_score, auroc = self.run_domains(chr, embed_rows)
            elif cfg.class_element == "Loop Domains":
                map, accuracy, f_score, auroc = self.run_loops(chr, embed_rows)
            elif cfg.class_element == "TADBs":
                map, accuracy, f_score, auroc = self.run_domains(chr, embed_rows)
            elif cfg.class_element == "subTADBs":
                map, accuracy, f_score, auroc = self.run_domains(chr, embed_rows)
            elif cfg.class_element == "Subcompartments":
                map, accuracy, f_score, auroc = self.run_sub_compartments(chr, embed_rows)

            map_frame = map_frame.append(
                {"chr": chr, "map": map, "fscore": f_score, "auroc": auroc, "accuracy": accuracy}, ignore_index=True)

        map_frame.to_csv(
            cfg.output_directory + "%s_metrics_%s_%s.csv" % (cfg.class_method, cfg.cell, cfg.class_element),
            sep="\t")

    def run_all_elements(self, model):
        """

        """
        for element in self.cfg.class_elements_list:
            self.cfg.class_element = element
            self.run_experiment(model)


if __name__ == '__main__':
    """
    Other mAP plots can be obtained by:
    Changing the cell type,
    The model associated with the cell type,
    Other models like Sniper and SCA.

    If you have all the data, you can use plot_combined function in analyses/plot/plot_fns.py
    """

    cfg = Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    "load model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    downstream_ob = DownstreamTasks(cfg)

    if cfg.class_run_elements:
        map = downstream_ob.run_experiment(model)
    elif cfg.class_run_all_elements:
        downstream_ob.run_all_elements(model)
