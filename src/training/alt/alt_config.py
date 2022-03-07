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
        self.chr_train_list = list(range(5, 6))
        self.chr_test_list = list(range(21, 22))
        self.save_processed_data = False

        "LSTM Paramters"
        self.hs_pos_lstm = 10
        self.sequence_length_pos = 100
        self.hs_mb_lstm = 4
        self.sequence_length_mb = 25
        self.hs_mega_lstm = 2
        self.sequence_length_mega = 10

        "MLP parameters"
        self.pos_embed_size = 16
        self.input_size_mlp = 2 * self.pos_embed_size
        self.hidden_size_fc1 = self.pos_embed_size
        self.hidden_size_fc2 = int(self.pos_embed_size / 2)
        self.output_size_mlp = 1
        self.mlp_batch_size = 650

        self.lstm_nontrain = False
        self.method = "hiclstm"

        "Hyperparameters"
        self.learning_rate = 0.01
        self.num_epochs = 500
        self.batch_size = 210
        self.max_norm = 10
        self.hic_smoothing = 8

        "Input Directories and file names"
        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.start_end_file = 'starts.npy'
        self.downstream_dir = "/data2/hic_lstm/downstream"
        self.model_name = "shuffle_new"

        "Output Directories"
        self.proj_dir = "/home/kevindsouza/Documents/projects/PhD/Hi-C-LSTM/"
        self.model_dir = self.proj_dir + 'saved_models/'
        self.output_directory = self.downstream_dir + "/predictions/"
        self.plot_dir = self.output_directory + 'data_plots/'
        self.processed_data_dir = self.output_directory + 'alt_processed_data/' + self.cell + "/"
        self.new_data_dir = self.output_directory + 'new_data/' + self.cell + "/"

        "create directories if they don't exist"
        for file_path in [self.model_dir, self.output_directory, self.plot_dir, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)


        "test"
        self.full_test = True
        self.get_zero_pred = False
