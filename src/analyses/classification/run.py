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


        self.fire_path = cfg.downstream_dir + "FIREs"
        self.loop_path = cfg.downstream_dir + "loops"
        self.domain_path = cfg.downstream_dir + "domains"
        self.subcompartment_path = cfg.downstream_dir + "subcompartments"
        self.chr_tad = 'chr' + str(chr)
        self.chr_ctcf = 'chr' + str(chr)
        self.chr_tad = 'chr' + str(chr)
        self.chr_fire = chr

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
            zero_features = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, mode="pos")

            if self.cfg.class_experiment == "baseline":
                zero_features = self.downstream_helper_ob.subc_baseline(zero_features, mode="pos")
            else:
                zero_features = self.downstream_helper_ob.merge_features_target(zero_features)

            features = features.append(zero_features)
            feature_matrix = features.copy()

        if self.cfg.compute_metrics:
            try:
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
        map, accuracy, f_score, auroc = self.run_xgboost(embed_rows, pe_chr, chr, zero_target=True, mode="ends")
        return map, accuracy, f_score, auroc

    def run_pe(self, cfg):
        logging.info("PE start")

        pe_ob = PeInteractions(cfg)
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_pe)

        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
            pe_window_labels = pe_data_chr_cell.filter(['window_start', 'window_end', 'label'], axis=1)
            pe_window_labels.rename(columns={'window_start': 'start', 'window_end': 'end', 'label': 'target'},
                                    inplace=True)
            pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            pe_window_labels = self.downstream_helper_ob.add_cum_pos(pe_window_labels, mode="ends")

            if self.exp == "baseline":
                feature_matrix = self.downstream_helper_ob.subc_baseline(Subcompartments, pe_window_labels, mode="ends")
            else:
                feature_matrix = self.downstream_helper_ob.get_feature_matrix(pe_window_labels)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

        return mean_map

    def run_fires(self, cfg):
        logging.info("fires start")

        fire_ob = Fires(cfg)
        fire_ob.get_fire_data(self.fire_path)
        fire_labeled = fire_ob.filter_fire_data(self.chr_fire)

        for cell in self.fire_cell_names:
            fire_window_labels = fire_labeled.filter(['start', 'end', cell + '_l'], axis=1)
            fire_window_labels.rename(columns={cell + '_l': 'target'}, inplace=True)
            fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            fire_window_labels = self.downstream_helper_ob.add_cum_pos(fire_window_labels, mode="ends")

            if self.exp == "baseline":
                feature_matrix = self.downstream_helper_ob.subc_baseline(Subcompartments, fire_window_labels,
                                                                         mode="ends")
            else:
                feature_matrix = self.downstream_helper_ob.get_feature_matrix(fire_window_labels)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

        return mean_map

    def run_loops(self, cfg):
        logging.info("loop start")

        for cell in self.loop_cell_names:
            loop_ob = Loops(cfg, cell, chr)
            loop_data = loop_ob.get_loop_data()

            col_list = ['x1', 'x2', 'y1', 'y2']
            zero_pos_frame = self.downstream_helper_ob.get_zero_pos(loop_data, col_list)

            feature_matrix = pd.DataFrame()
            for i in range(2):
                if i == 0:
                    temp_data = loop_data.rename(columns={'x1': 'start', 'x2': 'end'},
                                                 inplace=False)
                else:
                    temp_data = loop_data.rename(columns={'y1': 'start', 'y2': 'end'},
                                                 inplace=False)

                temp_data = temp_data.filter(['start', 'end', 'target'], axis=1)
                temp_data = temp_data.drop_duplicates(keep='first').reset_index(drop=True)
                temp_data = self.downstream_helper_ob.add_cum_pos(temp_data, mode="ends")
                if self.exp == "baseline":
                    features = self.downstream_helper_ob.subc_baseline(Subcompartments, temp_data, mode="ends")
                else:
                    features = self.downstream_helper_ob.get_feature_matrix(temp_data)

                feature_matrix = feature_matrix.append(features)

            zero_features = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, mode="pos")
            if self.exp == "baseline":
                zero_features = self.downstream_helper_ob.subc_baseline(Subcompartments, zero_features, mode="pos")
            else:
                zero_features = self.downstream_helper_ob.merge_features_target(zero_features)

            feature_matrix = feature_matrix.append(zero_features)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

        return mean_map

    def run_domains(self, cfg):
        logging.info("domain start")

        for cell in self.loop_cell_names:
            domain_ob = Domains(cfg, cell, chr)
            domain_data = domain_ob.get_domain_data()

            col_list = ['x1', 'x2']
            zero_pos_frame = self.downstream_helper_ob.get_zero_pos(domain_data, col_list)

            feature_matrix = pd.DataFrame()

            domain_data.rename(columns={'x1': 'start', 'x2': 'end'},
                               inplace=True)

            domain_data = domain_data.filter(['start', 'end', 'target'], axis=1)
            domain_data = domain_data.drop_duplicates(keep='first').reset_index(drop=True)
            domain_data = self.downstream_helper_ob.add_cum_pos(domain_data, mode="ends")
            if self.exp == "baseline":
                features = self.downstream_helper_ob.subc_baseline(Subcompartments, domain_data, mode="ends")
            else:
                features = self.downstream_helper_ob.get_feature_matrix(domain_data)

            feature_matrix = feature_matrix.append(features)

            zero_features = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, mode="pos")
            if self.exp == "baseline":
                zero_features = self.downstream_helper_ob.subc_baseline(Subcompartments, zero_features, mode="pos")
            else:
                zero_features = self.downstream_helper_ob.merge_features_target(zero_features)

            feature_matrix = feature_matrix.append(zero_features)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

        return mean_map

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

    def run_tss(self, cfg):
        logging.info("TSS start")
        pe_ob = PeInteractions(cfg)
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_pe)

        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
            pe_window_labels = pe_data_chr_cell.filter(['promoter_start', 'promoter_end', 'label'], axis=1)
            pe_window_labels.rename(columns={'promoter_start': 'pos', 'label': 'target'},
                                    inplace=True)
            pe_window_labels['pos_dup'] = pe_window_labels['pos']
            pe_window_labels = pe_window_labels.assign(target=1)
            pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            col_list = ['pos', 'pos_dup']
            zero_pos_frame = self.downstream_helper_ob.get_zero_pos(pe_window_labels, col_list)

            feature_matrix = pd.DataFrame()

            pe_window_labels = pe_window_labels[['pos', 'target']]
            pe_window_labels = self.downstream_helper_ob.add_cum_pos(pe_window_labels, mode="pos")
            if self.exp == "baseline":
                features = self.downstream_helper_ob.subc_baseline(Subcompartments, pe_window_labels, mode="pos")
            else:
                features = self.downstream_helper_ob.merge_features_target(pe_window_labels)
            feature_matrix = feature_matrix.append(features)

            zero_features = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, mode="pos")
            if self.exp == "baseline":
                zero_features = self.downstream_helper_ob.subc_baseline(Subcompartments, zero_features, mode="pos")
            else:
                zero_features = self.downstream_helper_ob.merge_features_target(zero_features)

            feature_matrix = feature_matrix.append(zero_features)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

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
                map, accuracy, f_score, auroc = self.run_tss(cfg)
            elif cfg.class_element == "PE-Interactions":
                map, accuracy, f_score, auroc = self.run_pe(cfg)
            elif cfg.class_element == "FIREs":
                map, accuracy, f_score, auroc = self.run_fires(cfg)
            elif cfg.class_element == "TADs":
                map, accuracy, f_score, auroc = self.run_domains(cfg)
            elif cfg.class_element == "subTADs":
                map, accuracy, f_score, auroc = self.run_domains(cfg)
            elif cfg.class_element == "Loop Domains":
                map, accuracy, f_score, auroc = self.run_loops(cfg)
            elif cfg.class_element == "TADBs":
                map, accuracy, f_score, auroc = self.run_domains(cfg)
            elif cfg.class_element == "subTADBs":
                map, accuracy, f_score, auroc = self.run_domains(cfg)
            elif cfg.class_element == "Subcompartments":
                map, accuracy, f_score, auroc = self.run_sub_compartments(cfg)

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
