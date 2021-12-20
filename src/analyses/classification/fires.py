import pandas as pd
import re


class Fires:
    def __init__(self, cfg):
        self.fire_data = None
        self.tad_data = None
        self.cfg = cfg

    def get_fire_data(self, fire_path):
        # fires = pd.read_excel(fire_path + '/FIRE_scores.xlsx', sheet_name="primary_cohort")
        # fires = fires.sort_values(by=['start']).reset_index(drop=True)

        fires = pd.read_pickle(fire_path + "/fires.pkl")

        ''' decimate by 25 to get the positions at 25 bp resolution '''
        fires["start"] = fires["start"] // self.cfg.resolution
        fires["end"] = fires["end"] // self.cfg.resolution

        fire_chosen = fires.iloc[:, 0:10]

        self.fire_data = fire_chosen

    def get_tad_data(self, tad_path, cell_names):

        tad_list = []
        for cell in cell_names:
            tads = pd.read_excel(tad_path + '/TAD_boundaries.xlsx', sheet_name=cell, names=["chr", "start", "end"])

            tads = tads.sort_values(by=['start']).reset_index(drop=True)

            ''' decimate by 25 to get the positions at 25 bp resolution '''
            tads["start"] = tads["start"] // self.cfg.resolution
            tads["end"] = tads["end"] // self.cfg.resolution

            tad_list.append(tads)

        self.tad_data = tad_list

    def filter_fire_data(self, chrom_fire):
        fire_data_chr = self.fire_data.loc[self.fire_data['chr'] == chrom_fire].reset_index(drop=True)

        fire_data_chr['GM12878_l'] = 0
        fire_data_chr['H1_l'] = 0
        fire_data_chr['IMR90_l'] = 0
        fire_data_chr['MES_l'] = 0
        fire_data_chr['MSC_l'] = 0
        fire_data_chr['NPC_l'] = 0
        fire_data_chr['TRO_l'] = 0

        fire_data_chr.loc[fire_data_chr['GM12878'] >= 0.5, 'GM12878_l'] = 1
        fire_data_chr.loc[fire_data_chr['H1'] >= 0.5, 'H1_l'] = 1
        fire_data_chr.loc[fire_data_chr['IMR90'] >= 0.5, 'IMR90_l'] = 1
        fire_data_chr.loc[fire_data_chr['MES'] >= 0.5, 'MES_l'] = 1
        fire_data_chr.loc[fire_data_chr['MSC'] >= 0.5, 'MSC_l'] = 1
        fire_data_chr.loc[fire_data_chr['NPC'] >= 0.5, 'NPC_l'] = 1
        fire_data_chr.loc[fire_data_chr['TRO'] >= 0.5, 'TRO_l'] = 1

        fire_labeled = fire_data_chr[
            ['chr', 'start', 'end', 'GM12878_l', 'H1_l', 'IMR90_l', 'MES_l', 'MSC_l', 'NPC_l', 'TRO_l']]

        return fire_labeled

    def filter_tad_data(self, chrom_tad):

        tad_list_filtered = []
        for i in range(len(self.tad_data)):
            tad_cell = self.tad_data[i]
            tad_data_chr = tad_cell.loc[tad_cell['chr'] == chrom_tad].reset_index(drop=True)

            tad_list_filtered.append(tad_data_chr)

        return tad_list_filtered

    def augment_tad_negatives(self, cfg, tad_df):

        neg_df = pd.DataFrame(columns=['start', 'end', 'target'])

        for i in range(tad_df.shape[0]):
            diff = tad_df.iloc[i]['end'] - tad_df.iloc[i]['start']

            start_neg = tad_df.iloc[i]['start'] - diff
            end_neg = tad_df.iloc[i]['start'] - 1

            if i == 0 or start_neg > tad_df.iloc[i - 1]['end']:
                neg_df = neg_df.append({'start': start_neg, 'end': end_neg, 'target': 0},
                                       ignore_index=True)

        tad_updated = pd.concat([tad_df, neg_df]).reset_index(drop=True)

        return tad_updated


if __name__ == '__main__':
    fire_path = "/opt/data/latent/data/downstream/FIREs"

    chrom_fire = 21
    chrom_tad = 'chr21'
    cell_names = ['GM12878', 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']

    fire_ob = Fires()
    fire_ob.get_fire_data(fire_path)
    fire_labeled = fire_ob.filter_fire_data(chrom_fire)

    fire_ob.get_tad_data(fire_path, cell_names)
    tad_filtered = fire_ob.filter_tad_data(chrom_tad)

    print("done")
