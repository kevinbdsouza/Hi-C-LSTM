from analyses.feature_attribution.tf import TFChip

# ig_pos_df = pd.DataFrame(columns=["sum_ig", "pos", "target"])
# ig_pos_df = downstream_ob.feature_importance(cfg, ig_pos_df, mode="cohesin")


def feature_importance(self, cfg, ig_pos_df, mode):
    ig_df = self.downstream_helper_ob.ig_rows[['sum_ig', 'pos']]

    if mode == "small":
        seg_ob = SegWay(cfg, self.chr_seg, self.segway_cell_names)
        annotations = seg_ob.segway_small_annotations()
        annotations = downstream_ob.downstream_helper_ob.get_window_data(annotations)
        annotations["pos"] = annotations["pos"] + downstream_ob.downstream_helper_ob.start
        ig_pos = pd.merge(ig_df, annotations, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "gbr":
        seg_ob = SegWay(cfg, self.chr_seg, self.segway_cell_names)
        annotations = seg_ob.segway_gbr().reset_index(drop=True)
        annotations = downstream_ob.downstream_helper_ob.get_window_data(annotations)
        annotations["pos"] = annotations["pos"] + downstream_ob.downstream_helper_ob.start
        ig_pos = pd.merge(ig_df, annotations, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "ctcf":
        cell = "GM12878"
        ctcf_ob = TFChip(cfg, cell, self.chr_ctcf)
        ctcf_data = ctcf_ob.get_ctcf_data()
        ctcf_data = ctcf_data.drop_duplicates(keep='first').reset_index(drop=True)
        ctcf_data = downstream_ob.downstream_helper_ob.get_window_data(ctcf_data)
        ctcf_data["pos"] = ctcf_data["pos"] + downstream_ob.downstream_helper_ob.start
        ig_pos = pd.merge(ig_df, ctcf_data, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "fire":
        fire_ob = Fires(cfg)
        fire_ob.get_fire_data(self.fire_path)
        fire_labeled = fire_ob.filter_fire_data(self.chr_fire)
        fire_window_labels = fire_labeled.filter(['start', 'end', "GM12878" + '_l'], axis=1)
        fire_window_labels.rename(columns={"GM12878" + '_l': 'target'}, inplace=True)
        fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

        fire_window_labels = downstream_ob.downstream_helper_ob.get_window_data(fire_window_labels)
        fire_window_labels["pos"] = fire_window_labels["pos"] + downstream_ob.downstream_helper_ob.start
        ig_pos = pd.merge(ig_df, fire_window_labels, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "tad":
        fire_ob = Fires(cfg)
        fire_ob.get_tad_data(self.fire_path, self.fire_cell_names)
        tad_cell = fire_ob.filter_tad_data(self.chr_tad)[0]
        tad_cell['target'] = 1
        tad_cell = tad_cell.filter(['start', 'end', 'target'], axis=1)
        tad_cell = tad_cell.drop_duplicates(keep='first').reset_index(drop=True)

        tad_cell = downstream_ob.downstream_helper_ob.get_window_data(tad_cell)
        tad_cell["pos"] = tad_cell["pos"] + downstream_ob.downstream_helper_ob.start

        ig_pos = pd.merge(ig_df, tad_cell, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "loops":
        loop_ob = Loops(cfg, "GM12878", chr)
        loop_data = loop_ob.get_loop_data()

        pos_matrix = pd.DataFrame()
        for i in range(2):
            if i == 0:
                temp_data = loop_data.rename(columns={'x1': 'start', 'x2': 'end'},
                                             inplace=False)
            else:
                temp_data = loop_data.rename(columns={'y1': 'start', 'y2': 'end'},
                                             inplace=False)

            temp_data = temp_data.filter(['start', 'end', 'target'], axis=1)
            temp_data = temp_data.drop_duplicates(keep='first').reset_index(drop=True)

            temp_data = downstream_ob.downstream_helper_ob.get_window_data(temp_data)
            temp_data["pos"] = temp_data["pos"] + downstream_ob.downstream_helper_ob.start
            pos_matrix = pos_matrix.append(temp_data)

        ig_pos = pd.merge(ig_df, pos_matrix, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "domains":
        domain_ob = Domains(cfg, "GM12878", chr)
        domain_data = domain_ob.get_domain_data()
        domain_data.rename(columns={'x1': 'start', 'x2': 'end'},
                           inplace=True)
        domain_data = domain_data.filter(['start', 'end', 'target'], axis=1)
        domain_data = domain_data.drop_duplicates(keep='first').reset_index(drop=True)

        domain_data = downstream_ob.downstream_helper_ob.get_window_data(domain_data)
        domain_data["pos"] = domain_data["pos"] + downstream_ob.downstream_helper_ob.start

        ig_pos = pd.merge(ig_df, domain_data, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    elif mode == "cohesin":
        cell = "GM12878"
        cohesin_df = pd.DataFrame()
        tf_ob = TFChip(cfg, cell, self.chr_ctcf)
        rad_data, smc_data = tf_ob.get_cohesin_data()
        rad_data = rad_data.drop_duplicates(keep='first').reset_index(drop=True)
        smc_data = smc_data.drop_duplicates(keep='first').reset_index(drop=True)

        rad_data = downstream_ob.downstream_helper_ob.get_window_data(rad_data)
        rad_data["pos"] = rad_data["pos"] + downstream_ob.downstream_helper_ob.start
        rad_data["target"] = "RAD21"
        cohesin_df = cohesin_df.append(rad_data)

        smc_data = downstream_ob.downstream_helper_ob.get_window_data(smc_data)
        smc_data["pos"] = smc_data["pos"] + downstream_ob.downstream_helper_ob.start
        smc_data["target"] = "SMC3"
        cohesin_df = cohesin_df.append(smc_data)

        ig_pos = pd.merge(ig_df, cohesin_df, on="pos")
        ig_pos.reset_index(drop=True, inplace=True)
        ig_pos_df = pd.concat([ig_pos_df, ig_pos], sort=True).reset_index(drop=True)

    return ig_pos_df
