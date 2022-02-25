import logging
import pandas as pd
import numpy as np
import os
import pickle
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class SegWay:
    def __init__(self, cfg, chr):
        self.segway_small_annotations_path = "/data2/hic_lstm/downstream/segway_small/"
        self.segway_small_file_name = "segway.hg19.bed"
        self.segway_small_label_file = "mnemonics.txt"
        self.segway_gbr_annotations_path = "/data2/hic_lstm/downstream/segway_gbr_domain/gbr_hg19_ann/"
        self.cfg = cfg
        self.chr = 'chr' + str(chr)
        self.seg_chr_bed = self.chr + '.bed'
        self.cell = cfg.cell
        self.segway_gbr_file_name = self.cell + ".bed"
        self.segway_gbr_chr_bed = self.chr + '.bed'

    def convert_to_bp_resolution(self, track):
        bp_track = np.zeros((25 * len(track, )))

        for i in range(len(track)):
            bp_track[25 * i:(i + 1) * 25 - 1] = track[i]

        return bp_track

    def create_signal_from_pickle(self, feature_path):
        signal = None

        for i in range(0, 24):
            features = pd.read_pickle(self.features_path)
            feature = features.loc[:, i]

            # bp_track = self.convert_to_bp_resolution(np.array(feature))

            for j in range(len(np.array(feature))):
                with open(self.dir + 'feature_' + str(i) + '.txt', "a") as myfile:
                    line = 'chr21   ' + str(25 * j + 1) + "   " + str(25 * (j + 1)) + "    " + str(feature[j]) + '\n'

                    myfile.write(line)

        return signal

    def load_features(self, chr_len):

        for i in range(11, 24):
            feature_path = self.dir + 'feature_' + str(i) + '.npy'
            feature = np.load(feature_path)
            feature = feature[:chr_len]

            with open(self.dir + 'feature_' + str(i) + '.wigFix', "ab") as myfile:
                np.savetxt(myfile, feature, delimiter=',', newline='\n')

        return

    def rename_files(self):
        for filename in os.listdir(self.dir):
            os.rename(filename, filename)

        pass

    def run_genome_euclid(self):
        target = open(
            '/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/lstm_features/feat_chr_21.pkl',
            'rb')
        chr_df = pickle.load(target)
        target.close()

        chr_df = chr_df.drop(['target', 'gene_id'], axis=1)

        all_means = []
        for i in range(1, len(chr_df) - 1):
            all_dst = []
            diff = i
            for k in range(0, len(chr_df) - 1):
                if k + diff >= len(chr_df):
                    break
                a = chr_df.iloc[k, :]
                b = chr_df.iloc[k + diff, :]
                all_dst.append(distance.euclidean(a, b))
            all_means.append(np.mean(all_dst))

        return all_means

    def run_segway(self):

        GENOMEDATA_DIRNAME = "/home/kevindsouza/Documents/projects/latentGenome/results/04-27-2019_n/h_110/5e-14/21/lstm_features/genomedata/genomedata.test"

        # run.main(["--random-starts=3", "train", GENOMEDATA_DIRNAME])

        return

    def segway_small_annotations(self):
        """

        """
        column_list = ["chr", "start", "end", "target", "num", "dot", "start_2", "end_2", "color"]
        col_numbers = list(range(0, len(column_list)))
        new_df = pd.DataFrame()

        segway_small_annotations = pd.read_csv(self.segway_small_annotations_path + self.seg_chr_bed,
                                               sep="\t", engine='python', columns=column_list)
        col_series = pd.Series(segway_small_annotations.columns)
        new_df = new_df.append(col_series, ignore_index=True)
        segway_small_annotations.columns = col_numbers
        segway_small_new_annotations = pd.concat([new_df, segway_small_annotations])
        segway_small_new_annotations.columns = column_list
        segway_small_new_annotations = segway_small_new_annotations[['start', 'end', 'target']]
        segway_small_new_annotations["start"] = segway_small_new_annotations["start"].astype(int) // self.cfg.resolution
        segway_small_new_annotations["end"] = segway_small_new_annotations["end"].astype(int) // self.cfg.resolution

        return segway_small_new_annotations.reset_index(drop=True)

    def segway_gbr(self):

        column_list = ["chr", "start", "end", "target", "num", "dot", "start_2", "end_2", "color"]
        col_numbers = list(range(0, len(column_list)))
        new_df = pd.DataFrame()
        segway_gbr_annotations = pd.read_csv(
            self.segway_gbr_annotations_path + self.cell + "/" + self.segway_gbr_chr_bed,
            sep="\t", engine='python')

        col_series = pd.Series(segway_gbr_annotations.columns)
        new_df = new_df.append(col_series, ignore_index=True)
        segway_gbr_annotations.columns = col_numbers
        segway_gbr_new_annotations = pd.concat([new_df, segway_gbr_annotations])
        segway_gbr_new_annotations.columns = column_list
        segway_gbr_new_annotations = segway_gbr_new_annotations[['start', 'end', 'target']]
        segway_gbr_new_annotations["start"] = segway_gbr_new_annotations["start"].astype(int) // self.cfg.resolution
        segway_gbr_new_annotations["end"] = segway_gbr_new_annotations["end"].astype(int) // self.cfg.resolution

        return segway_gbr_new_annotations


if __name__ == '__main__':
    cfg = None
    chr = 'chr21'
    cell = "GM12878"

    seg_ob = SegWay(cfg, chr, cell)
    fasta_seq = seg_ob.read_fasta(seg_ob.fasta_path)

    chr_len = len(fasta_seq)

    # signal = seg_ob.create_signal_from_pickle(seg_ob.features_path)
    # seg_ob.load_features(chr_len)

    # seg_ob.run_segway()
    all_means = seg_ob.run_genome_euclid()
    print("done")
