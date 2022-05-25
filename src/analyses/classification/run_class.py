from analyses.classification.rna_seq import GeneExp
from training.config import Config
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
from analyses.classification.downstream_helper import DownstreamHelper
from analyses.classification.fires import Fires
from analyses.classification.rep_timing import Rep_timing
from analyses.classification.pe_interactions import PeInteractions
from analyses.classification.loops import Loops
from analyses.classification.domains import Domains
from analyses.classification.subcompartments import Subcompartments
from analyses.feature_attribution.tf import TFChip
from analyses.classification.model_class import MultiClass
from training.model import SeqLSTM
import torch
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1


class DownstreamTasks:
    """
    Class to run all downstream classifi experiments using XGBoost.
    Compute mAP, Accuaracy, PR curves, AuROC, and F-score.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_model_dir = cfg.model_dir
        self.calculate_map = True
        self.downstream_helper_ob = DownstreamHelper(cfg)
        self.df_columns = [str(i) for i in range(0, 16)] + ["pos"]
        self.class_columns = [str(i) for i in range(0, 10)]
        self.iter = 0

    def get_zero_one(self, window_labels, chr, zero_target, mode, combined):
        window_labels = window_labels.drop_duplicates(keep='first').reset_index(drop=True)

        if zero_target:
            if mode == "ends":
                col_list = ['start', 'end']
            else:
                col_list = ['pos']
            zero_pos_frame = self.downstream_helper_ob.get_zero_pos(window_labels, col_list, chr)
            zero_pos_frame = self.downstream_helper_ob.add_cum_pos(zero_pos_frame, chr, mode="pos")

            window_labels = self.downstream_helper_ob.add_cum_pos(window_labels, chr, mode=mode)
            if mode == "ends":
                window_labels = self.downstream_helper_ob.get_pos_data(window_labels, chr)
        else:
            zero_pos_frame = None
            if not combined:
                window_labels = self.downstream_helper_ob.add_cum_pos(window_labels, chr, mode=mode)

                zero_pos_frame = window_labels.loc[window_labels["target"] == 0]
                window_labels = window_labels.loc[window_labels["target"] == 1]

                if mode == "ends":
                    zero_pos_frame = self.downstream_helper_ob.get_pos_data(zero_pos_frame, chr)
                    window_labels = self.downstream_helper_ob.get_pos_data(window_labels, chr)

        return window_labels, zero_pos_frame

    def run_gene_expression(self, chr):
        """
        run_gene_expression(chr, embed_rows) -> float, float, float, float
        Gets gene expression data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        rna_seq_ob = GeneExp(self.cfg, chr)
        rna_seq_ob.get_rna_seq()
        rna_seq_chr = rna_seq_ob.filter_rna_seq()
        return rna_seq_chr

    def run_rep_timing(self, chr):
        """
        run_rep_timing(chr, embed_rows) -> float, float, float, float
        Gets replication timing data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        rep_ob = Rep_timing(self.cfg, chr)
        rep_chr = rep_ob.get_rep_data()
        return rep_chr

    def run_enhancers(self, chr):
        """
        run_enhancers(chr, embed_rows) -> float, float, float, float
        Gets Enhancer data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)
        return pe_chr

    def run_tss(self, chr):
        """
        run_tss(chr, embed_rows) -> float, float, float, float
        Gets TSS data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)
        return pe_chr

    def run_pe(self, chr):
        """
        run_pe(chr, embed_rows) -> float, float, float, float
        Gets PE interaction data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        pe_ob = PeInteractions(cfg, chr)
        pe_chr = pe_ob.get_pe_data()
        pe_chr = pe_ob.filter_pe_data(pe_chr)
        return pe_chr

    def run_fires(self, chr):
        """
        run_fires(chr, embed_rows) -> float, float, float, float
        Gets FIRE data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        fire_ob = Fires(cfg, chr, mode="classifi")
        fire_ob.get_fire_data()
        fire_chr = fire_ob.filter_fire_data()
        return fire_chr

    def run_domains(self, chr):
        """
        run_domains(chr, embed_rows) -> float, float, float, float
        Gets TAD, subTAD, and boundaries data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """
        domain_ob = Domains(cfg, chr, mode="class")

        "get data and run xgboost according to type of domain element"
        if self.cfg.class_element == "TADs":
            data = domain_ob.get_tad_data()
        elif self.cfg.class_element == "subTADs":
            data = domain_ob.get_tad_data()
        elif self.cfg.class_element == "TADBs":
            tf_ob = TFChip(cfg, chr)
            data = domain_ob.get_tad_boundaries(tf_ob, ctcf="all")
        elif self.cfg.class_element == "subTADBs":
            data = domain_ob.get_subtad_boundaries()

        return data

    def run_loops(self, chr):
        """
        run_loops(chr, embed_rows) -> float, float, float, float
        Gets loop data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        loop_ob = Loops(cfg, chr, mode="classifi")
        loop_chr = loop_ob.get_loop_data()
        return loop_chr

    def run_sub_compartments(self, chr):
        """
        run_sub_compartments(chr, embed_rows) -> float, float, float, float
        Gets subcompartment data for given cell type and chromosome.
        Runs xgboost using representations from chosen method and celltype. Or runs baseline.
        Returns classifi metrics.
        Args:
            chr (int): chromosome to run classifi on.
            embed_rows (DataFrame): Dataframe with representations and positions.
        """

        sc_ob = Subcompartments(cfg, chr)
        sc_chr = sc_ob.get_sc_data()
        return sc_chr

    def return_data_for_element(self, element, chr):
        self.cfg.class_element = element

        if element == "Gene Expression":
            data = self.run_gene_expression(chr)
            model_base = "gene"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=False, mode="ends", combined=False)
        elif element == "Replication Timing":
            data = self.run_rep_timing(chr)
            model_base = "rep"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=False, mode="ends", combined=False)
        elif element == "Enhancers":
            data = self.run_enhancers(chr)
            model_base = "en"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "TSS":
            data = self.run_tss(chr)
            model_base = "tss"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "PE-Interactions":
            data = self.run_pe(chr)
            model_base = "pei"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "FIREs":
            data = self.run_fires(chr)
            model_base = "fire"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=False, mode="ends", combined=False)
        elif element == "TADs":
            data = self.run_domains(chr)
            model_base = "tad"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "subTADs":
            data = self.run_domains(chr)
            model_base = "subtad"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "Loop Domains":
            data = self.run_loops(chr)
            model_base = "loop"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="ends", combined=False)
        elif element == "TADBs":
            data = self.run_domains(chr)
            model_base = "tadb"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="pos", combined=False)
        elif element == "subTADBs":
            data = self.run_domains(chr)
            model_base = "subtadb"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=True, mode="pos", combined=False)
        elif element == "Subcompartments":
            data = self.run_sub_compartments(chr)
            model_base = "subc"
            one_data, zero_data = self.get_zero_one(data, chr, zero_target=False, mode="ends", combined=False)

        return one_data, zero_data, model_base

    def run_multi_label(self, cfg):
        """
        run_multi_label(cfg) -> No return object
        Runs xgboost for the given model and all classifi elements.
        Uses provide configuration. Saves classifi metrics in CSV file.
        Args:
            cfg (Config): Model whose representations need to be used to run xgboost.
        """

        self.cfg.class_model_mode = "test"

        for chr in cfg.chr_train_list:
            print("Chromosome: {}".format(chr))

            self.cfg.model_name = "class_chr" + str(chr)
            self.cfg.chr = chr

            "load class model"
            model = MultiClass(self.cfg, device).to(device)
            model.load_weights()

            "load embed model"
            embed_model = SeqLSTM(self.cfg, device).to(device)
            embed_model.load_weights()

            "Initalize optimizer"
            optimizer = model.compile_optimizer()

            "Set up Tensorboard logging"
            timestr = time.strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter('./tensorboard_logs/' + self.cfg.model_name + timestr)

            try:
                main_data = pd.read_csv(cfg.output_directory + "element_data_chr%s.csv" % (chr))
            except:
                main_data = pd.DataFrame()
                for i, element in enumerate(cfg.class_elements_list):
                    print("Data for: %s" % (element))
                    element_data, _, _ = self.return_data_for_element(element, chr, )
                    element_data[self.class_columns] = 0
                    element_data[self.cfg.class_dict[element]] = 1
                    element_data = element_data[self.class_columns + ["pos"]]

                    if i == 0:
                        main_data = pd.concat([main_data, element_data])
                    else:
                        common_pos = element_data[element_data["pos"].isin(main_data["pos"])]
                        diff_pos = element_data[~element_data["pos"].isin(main_data["pos"])]

                        common_pos = common_pos.drop_duplicates(keep='first').reset_index(drop=True)
                        main_data.loc[main_data["pos"].isin(common_pos["pos"]), self.cfg.class_dict[element]] = 1

                        main_data = pd.concat([main_data, diff_pos])
                        main_data = main_data.drop_duplicates(keep='first').reset_index(drop=True)

                main_data = main_data.sample(frac=1)

            if self.cfg.class_model_mode == "train":
                iter = 0
                for epoch in range(self.cfg.num_epochs):
                    iter, model, loss = model.train_model_multi(embed_model, epoch, optimizer, writer, main_data, iter,
                                                                self.cfg)

                print('Final loss for chromosome %s : %s' % (chr, loss))
                main_data.to_csv(cfg.output_directory + "element_data_chr%s.csv" % (chr))
            else:
                map, _ = model.test_model_multi(main_data, self.cfg, embed_model)

                print('Mean mAP for chromosome %s : %s' % (chr, map))


if __name__ == '__main__':
    """
    Script to run xgboost for representations from class_method specified in config. One of hiclstm, sniper, and sca.
    Baselines like pca_baseline, subc_baseline can be specified in class_experiment in config. 
    If class_experiment is not specified as baseline, assumes experiment is based on one of the methods. 
    Sets class_experiment as class_method. 
    All experiments except for Subcompartments are binary. Subcompartments is multiclass.
    To change the cell type, specify the cell in config. 
    The appropriate model and elements for cell type will be loaded.
    """

    cfg = Config()
    cell = cfg.cell

    "downstream data object"
    downstream_ob = DownstreamTasks(cfg)

    "run multilabel"
    downstream_ob.run_multi_label(cfg)
