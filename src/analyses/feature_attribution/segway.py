import logging
import pandas as pd
import numpy as np
import os
import pickle
from scipy.spatial import distance
from training.config import Config


class SegWay:
    def __init__(self, cfg, chr):
        self.segway_small_annotations_path = "/data2/hic_lstm/downstream/segway_small/"
        self.segway_small_file_name = "segway.hg19.bed"
        self.segway_label_file = self.segway_small_annotations_path + "mnemonics.txt"
        self.segway_gbr_annotations_path = "/data2/hic_lstm/downstream/segway_gbr_domain/gbr_hg19_ann/"
        self.cfg = cfg
        self.chr = 'chr' + str(chr)
        self.seg_chr_bed = self.chr + '.bed'
        self.cell = cfg.cell
        self.segway_gbr_file_name = self.cell + ".bed"
        self.segway_gbr_chr_bed = self.chr + '.bed'

    def convert_segway_labels(self, segway_annotations):
        """
        convert_segway_labels(segway_annotations) -> DataFrame
        Converts numerical segway labels to string segway labels.
        Args:
            segway_annotations (DataFrame): Segway data to be converted
        """

        labels = pd.read_csv(self.segway_label_file, sep="\t")
        for i in range(len(labels)):
            segway_annotations.loc[segway_annotations["target"] == labels.loc[i]["old"], "target"] = labels.loc[i][
                "description"]

        segway_annotations = segway_annotations.loc[segway_annotations["target"] != "H3K9me1 only"]
        return segway_annotations.reset_index(drop=True)

    def segway_small_annotations(self):
        """
        segway_small_annotations() -> DataFrame
        Obtains segway small scale positions and labels.
        Args:
            NA
        """
        column_list = ["chr", "start", "end", "target", "num", "dot", "start_2", "end_2", "color"]
        segway_annotations = pd.read_csv(self.segway_small_annotations_path + self.seg_chr_bed,
                                         sep="\t", engine='python', header=None)
        segway_annotations.columns = column_list
        segway_annotations = segway_annotations[['start', 'end', 'target']]
        segway_annotations["start"] = segway_annotations["start"].astype(int) // self.cfg.resolution
        segway_annotations["end"] = segway_annotations["end"].astype(int) // self.cfg.resolution

        segway_annotations = self.convert_segway_labels(segway_annotations)
        return segway_annotations

    def convert_gbr_labels(self, gbr_annotations):
        """
        convert_gbr_labels(gbr_annotations) -> DataFrame
        Converts segway GBR labels to better string segway labels.
        Currently not implemented.
        Args:
            gbr_annotations (DataFrame): GBR data to be converted
        """
        return gbr_annotations.reset_index(drop=True)

    def segway_gbr(self):
        """
        segway_gbr() -> DataFrame
        Obtains segway GBR domain scale positions and labels.
        Args:
            NA
        """
        column_list = ["chr", "start", "end", "target", "num", "dot", "start_2", "end_2", "color"]
        gbr_annotations = pd.read_csv(
            self.segway_gbr_annotations_path + self.cell + "/" + self.segway_gbr_chr_bed,
            sep="\t", engine='python', header=None)
        gbr_annotations.columns = column_list
        gbr_annotations = gbr_annotations[['start', 'end', 'target']]
        gbr_annotations["start"] = gbr_annotations["start"].astype(int) // self.cfg.resolution
        gbr_annotations["end"] = gbr_annotations["end"].astype(int) // self.cfg.resolution

        gbr_annotations = self.convert_gbr_labels(gbr_annotations)
        return gbr_annotations


if __name__ == '__main__':
    cfg = Config()
    chr = 21
    cell = cfg.cell
    seg_ob = SegWay(cfg, chr)
