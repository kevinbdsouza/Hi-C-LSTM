import logging
from analyses.classification.rna_seq import RnaSeq
from training.config import Config
import pandas as pd
from analyses.classification.downstream_helper import DownstreamHelper
from analyses.classification.fires import Fires
from analyses.classification.rep_timing import Rep_timing
from analyses.classification.pe_interactions import PeInteractions
from analyses.classification.loops import Loops
from analyses.classification.domains import Domains
from analyses.classification.subcompartments import Subcompartments

logger = logging.getLogger(__name__)


class DownstreamTasks:
    def __init__(self, cfg, chr, mode):
        self.data_dir = "/data2/hic_lstm/downstream/"
        self.rna_seq_path = self.data_dir + "RNA-seq"
        self.pe_int_path = self.data_dir + "PE-interactions"
        self.fire_path = self.data_dir + "FIREs"
        self.rep_timing_path = self.data_dir + "replication_timing"
        self.loop_path = self.data_dir + "loops"
        self.domain_path = self.data_dir + "domains"
        self.phylo_path = self.data_dir + "phylogenetic_scores"
        self.subcompartment_path = self.data_dir + "subcompartments"
        self.fire_cell_names = ['GM12878']  # , 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']
        self.rep_cell_names = ['IMR90']  # , 'HUVEK', 'K562']
        self.chr = chr
        self.pe_cell_names = ['E116']  # , 'E117', 'E123', 'E017']
        self.loop_cell_names = ['GM12878']  # , 'IMR90', 'HUVEC', 'HMEC', 'K562', 'HeLa']
        self.sbc_cell_names = ['GM12878']
        self.chr_rna = str(chr)
        self.chr_pe = 'chr' + str(chr)
        self.chr_tad = 'chr' + str(chr)
        self.chr_rep = 'chr' + str(chr)
        self.chr_ctcf = 'chr' + str(chr)
        self.chr_tad = 'chr' + str(chr)
        self.chr_fire = chr
        self.saved_model_dir = cfg.model_dir
        self.calculate_map = True
        self.exp = "map"
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode)

    def run_rna_seq(self, cfg):
        logging.info("RNA-Seq start")

        rna_seq_ob = RnaSeq(cfg)
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_rna)

        rna_seq_chr['target'] = 0

        # 1,58
        for col in range(49, 50):
            rna_seq_chr.loc[rna_seq_chr.iloc[:, col] >= 0.5, 'target'] = 1
            rna_window_labels = rna_seq_chr.filter(['start', 'end', 'target'], axis=1)
            rna_window_labels = rna_window_labels.drop_duplicates(keep='first').reset_index(drop=True)
            rna_window_labels = rna_window_labels.drop([410, 598]).reset_index(drop=True)

            rna_window_labels = self.downstream_helper_ob.add_cum_pos(rna_window_labels, mode="ends")

            if self.exp == "baseline":
                feature_matrix = self.downstream_helper_ob.subc_baseline(Subcompartments, rna_window_labels,
                                                                         mode="ends")
            else:
                feature_matrix = self.downstream_helper_ob.get_feature_matrix(rna_window_labels)

            logging.info("chr : {} - cell : {}".format(str(self.chr), rna_seq_chr.columns[col]))

            if feature_matrix.empty:
                continue

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map(feature_matrix, mode="binary", exp=self.exp)

        return mean_map

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

    def run_rep_timings(self, cfg):
        logging.info("rep start")

        rep_ob = Rep_timing(cfg)
        rep_ob.get_rep_data(self.rep_timing_path, self.rep_cell_names)
        rep_filtered = rep_ob.filter_rep_data(self.chr_rep)

        for i, cell in enumerate(self.rep_cell_names):

            rep_data_cell = rep_filtered[i]
            rep_data_cell = rep_data_cell.filter(['start', 'end', 'target'], axis=1)
            rep_data_cell = rep_data_cell.drop_duplicates(keep='first').reset_index(drop=True)

            rep_data_cell = self.downstream_helper_ob.add_cum_pos(rep_data_cell, mode="ends")

            if self.exp == "baseline":
                feature_matrix = self.downstream_helper_ob.subc_baseline(Subcompartments, rep_data_cell, mode="ends")
            else:
                feature_matrix = self.downstream_helper_ob.get_feature_matrix(rep_data_cell)

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

    def run_p_and_e(self, cfg):
        logging.info("P and E start")
        pe_ob = PeInteractions(cfg)
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_pe)

        element_list = ["enhancer"]
        for e in element_list:
            for cell in self.pe_cell_names:
                pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
                pe_window_labels = pe_data_chr_cell.filter([e + '_start', e + '_end', 'label'], axis=1)
                pe_window_labels.rename(columns={e + '_start': 'start', e + '_end': 'end', 'label': 'target'},
                                        inplace=True)
                pe_window_labels = pe_window_labels.assign(target=1)
                pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

                col_list = ['start', 'end']
                zero_pos_frame = self.downstream_helper_ob.get_zero_pos(pe_window_labels, col_list)

                feature_matrix = pd.DataFrame()

                pe_window_labels = self.downstream_helper_ob.add_cum_pos(pe_window_labels, mode="ends")
                if self.exp == "baseline":
                    features = self.downstream_helper_ob.subc_baseline(Subcompartments, pe_window_labels, mode="ends")
                else:
                    features = self.downstream_helper_ob.get_feature_matrix(pe_window_labels)
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


if __name__ == '__main__':

    cfg = Config()
    test_chr = list(range(22, 23))
    map_frame = pd.DataFrame(
        columns=["chr", "gene_map", "pe_map", "fire_map", "rep_map", "loops_map", "domains_map", "enhancers_map",
                 "tss_map"])

    for chr in test_chr:
        logging.info("Downstream start Chromosome: {}".format(chr))

        downstream_ob = DownstreamTasks(cfg, chr, mode='lstm')

        gene_map = downstream_ob.run_rna_seq(cfg)

        pe_map = downstream_ob.run_pe(cfg)

        fire_map = downstream_ob.run_fires(cfg)

        rep_map = downstream_ob.run_rep_timings(cfg)

        loops_map = downstream_ob.run_loops(cfg)

        domains_map = downstream_ob.run_domains(cfg)

        # mapdict_subcomp = downstream_ob.run_sub_compartments(cfg)

        enhancers_map = downstream_ob.run_p_and_e(cfg)

        tss_map = downstream_ob.run_tss(cfg)

        map_frame = map_frame.append(
            {"chr": chr, "gene_map": gene_map, "pe_map": pe_map, "fire_map": fire_map, "rep_map": rep_map,
             "loops_map": loops_map, "domains_map": domains_map, "enhancers_map": enhancers_map, "tss_map": tss_map},
            ignore_index=True)

    map_frame.to_csv(cfg.output_directory + "mapframe_%s.csv" % (cfg.cell), sep="\t")
    print("done")
