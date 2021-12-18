import os
import pathlib


class Config:
    def __init__(self):

        ##########################################
        ############ Data Parameters #############
        ##########################################

        self.num_chr = 23
        self.genome_len = 288091
        self.resolution = 10000

        ##########################################
        ############ Model Parameters ############
        ##########################################

        self.pos_embed_size = 16
        self.input_size_lstm = 2 * self.pos_embed_size
        self.hidden_size_lstm = 8
        self.output_size_lstm = 1
        self.sequence_length = 150  # length of each input sequence

        self.distance_cut_off_mb = int(self.sequence_length / 2)
        # for dense input matrix, takes sequence_length around diagonal for each row

        ##########################################
        ############# Hyperparameters ############
        ##########################################

        self.learning_rate = 0.01
        self.num_epochs = 40
        self.batch_size = 210
        self.max_norm = 10  # for gradient clipping
        self.lstm_nontrain = False

        ##########################################
        ############ Input Directories ###########
        ##########################################

        self.cell = "GM12878"
        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.start_end_file = 'starts.npy'
        self.downstream_dir = "/data2/hic_lstm/downstream"

        ##########################################
        ############ Output Locations ############
        ##########################################

        self.model_dir = '../saved_models/'
        self.output_directory = self.downstream_dir + "/predictions/"
        self.plot_dir = self.output_directory + 'data_plots/'
        self.processed_data_dir = self.output_directory + 'processed_data/' + self.cell + "/"

        for file_path in [self.model_dir, self.output_directory, self.plot_dir, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)
