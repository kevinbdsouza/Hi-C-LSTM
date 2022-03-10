import os
import pathlib


class Config:
    def __init__(self):
        """
        Includes Data Parameters, Model Parameters, Hyperparameters, Input Directories
        File Names, Model Names, Output Directories
        """

        "Data Parameters"
        self.num_chr = 23
        self.genome_len = 288091
        self.resolution = 10000
        self.cell = "GM12878"
        self.chr_train_list = list(range(2, 23))
        self.chr_test_list = list(range(22, 23))
        self.save_processed_data = False

        "Model Paramters"
        self.pos_embed_size = 16
        self.input_size_lstm = 2 * self.pos_embed_size
        self.hidden_size_lstm = 8
        self.output_size_lstm = 1
        self.sequence_length = 150
        self.distance_cut_off_mb = int(self.sequence_length / 2)
        self.lstm_nontrain = False
        self.window_model = False
        self.method = "hiclstm"

        "Hyperparameters"
        self.learning_rate = 0.01
        self.num_epochs = 40
        self.batch_size = 210
        self.max_norm = 10
        self.hic_smoothing = 8

        "Input Directories and file names"
        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.schic_reads_file = "/GSM2254215_ML1.percentages.txt"
        self.schic_pairs_file = "/GSM2254215_ML1.validPairs.txt"
        self.start_end_file = 'starts.npy'
        self.downstream_dir = "/data2/hic_lstm/downstream"
        self.model_name = "shuffle_" + self.cell

        "decoder parameters"
        self.decoder_name = "hiclstm_fc"
        self.decoder = "fc"
        self.compute_pca = False
        self.get_zero_pred = False
        self.full_test = True
        self.recon_compute_representation = False
        self.recon_save_representation = False
        self.train_decoders = False
        self.test_decoders = True
        self.recon_get_predictions = False
        self.dec_learning_rate = 0.01
        self.dec_max_norm = 10
        self.decoder_epochs = 20
        self.decoder_train_list = [21, 19]
        self.decoder_test_list = list(range(22, 23))

        "Output Directories"
        self.proj_dir = "/home/kevindsouza/Documents/projects/PhD/Hi-C-LSTM/"
        self.model_dir = self.proj_dir + 'saved_models/'
        self.output_directory = self.downstream_dir + "/predictions/"
        self.plot_dir = self.output_directory + 'data_plots/'
        self.processed_data_dir = self.output_directory + 'processed_data/' + self.cell + "/"
        self.new_data_dir = self.output_directory + 'new_data/' + self.cell + "/"

        "create directories if they don't exist"
        for file_path in [self.model_dir, self.output_directory, self.plot_dir, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)

        "ctcf dots"
        self.ctcf_dots_threshold = 41

        "captum"
        self.run_captum = False
        self.tf_file_path = self.output_directory + "tf_db.csv"
        self.tf_columns = ["HGNC symbol", "chromosome", "start"]
        self.ig_run_tfs = True
        self.ig_run_chip = True
        self.ig_run_elements = False
        self.ig_element = "Segway"
        self.ig_run_all_elements = False
        self.ig_elements_list = ["CTCF", "RAD21", "SMC3", "GBR", "TADs", "FIREs", "Domains", "Loop_Domains",
                                 "Merge_Domains", "TADBs", "TADBsCTCF+", "TADBsCTCF-", "Loop_CTCFCohesin",
                                 "NonLoop_CTCFCohesin"]

        "classification"
        if self.cell == "GM12878":
            self.class_elements_list = ["Gene Expression", "Replication Timing", "Enhancers", "TSS",
                                        "PE-Interactions", "FIREs", "TADs", "subTADs", "Loop Domains",
                                        "TADBs", "subTADBs", "Subcompartments"]
            self.metric_frame_columns = self.class_elements_list.append("chr")
        else:
            self.class_elements_list = ["Gene Expression", "Enhancers", "TSS", "TADs", "subTADs",
                                        "Loop Domains", "TADBs", "subTADBs", "Subcompartments"]
            self.metric_frame_columns = self.class_elements_list.append("chr")
        self.class_compute_representation = False
        self.class_element = "Gene Expression"
        self.class_method = "pca"
        self.class_run_elements = True
        self.class_run_all_elements = False
        self.class_pr = False
        self.class_mode = "binary"
        self.compute_metrics = True
        self.n_folds = 3
        self.regression_mode = "mlp"

        "knockout"
        self.ko_compute_test = False
        self.ko_experiment = "tadbs"
        self.ctcf_indices = "all"
        self.foxg1_chr = [14]
        self.ctcf_indices_21 = [279219, 279229]
        self.ctcf_indices_22 = [284706, 284743]
        self.foxg1_indices = [222863]
        self.ko_window = 10
        self.ko_increment = 10
        self.ko_mode = "shift"
        self.perform_ko = False
        self.load_ko = False
        self.compare_ko = False
        self.save_kopred = False
        self.compute_avg_diff = False
        self.run_tal = True
        self.tal_pre = False
        self.tal_train = False
        self.tal_test = True
        self.tal_mode = "tal1_ko"
        self.compare_tal = True
        self.check_ko = True
        self.tal_pre_ko = False
        self.tal_plot_wt = True
        self.tal_plot_ko = True
        self.lmo2_chr = 11

        "duplication"
        self.melo_insert = True
        self.compare_dup = True
        self.dupl_load_data = False
        self.dupl_compute_test = False
        self.chunk_start = 256803
        self.chunk_end = 257017
        self.chunk_start_rep = 6794
        self.chunk_end_rep = 7008
        self.dupl_start = 257018
        self.dupl_start_rep = 7009
        self.dupl_end = 257232
        self.dupl_end_rep = 7223
        self.melo_shift = 215
        self.plt_start = 6700
        self.plt_end = 7440
        self.dupl_chrs = [17]
        self.dupl_mode = "fusion"
