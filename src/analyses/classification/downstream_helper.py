import logging
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from training.config import Config
from training.data_utils import get_cumpos
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import *
import seaborn as sn
#import matplotlib as mpl
#mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DownstreamHelper:
    def __init__(self, cfg, chr, mode):
        self.cfg = cfg
        self.chr = chr
        self.cell = cfg.cell
        self.start_ends = np.load(cfg.hic_path + cfg.start_end_file, allow_pickle=True).item()
        self.start = self.start_ends["chr" + str(chr)]["start"] + get_cumpos(cfg, chr)
        self.stop = self.start_ends["chr" + str(chr)]["stop"] + get_cumpos(cfg, chr)
        self.feature_columns = list(range(0, 16))
        self.chr_len = cfg.genome_len
        self.num_subc = 5
        self.embed_rows = None
        self.pred_rows = None

        if mode == 'sniper':
            self.embed_path = self.cfg.sniper_annotation_path + "encoding_" + str(self.chr) + ".npy"
            self.embed_rows = pd.DataFrame(np.load(self.embed_path))
            self.embed_rows["pos"] = self.embed_rows.index + self.start
        elif mode == 'lstm':
            self.embed_rows = pd.DataFrame(columns=list(np.arange(cfg.pos_embed_size)) + ["pos"])
            self.pred_rows = pd.read_csv(
                cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (self.cell, str(chr)), sep="\t")
            self.pred_rows = self.pred_rows.drop(['Unnamed: 0'], axis=1)
            for k in range(cfg.pos_embed_size):
                self.embed_rows[k] = self.pred_rows[str(k)]
            self.embed_rows["pos"] = self.pred_rows["i"]
        elif mode == "graph":
            self.embed_path = self.cfg.graph_annotation_path + "embedding_" + str(self.chr) + ".npy"
            self.embed_rows = pd.DataFrame(np.load(self.embed_path))
            self.embed_rows["pos"] = self.embed_rows.index + self.start
        elif mode == "pca":
            self.embed_path = self.cfg.pca_annotation_path + "encoding_" + str(self.chr) + ".npy"
            self.embed_rows = pd.DataFrame(np.load(self.embed_path))
            self.embed_rows["pos"] = self.embed_rows.index + self.start

    def cat_convert(self, y_test, y_valid, feature_matrix):

        categorical_convert = LabelEncoder()

        y_cols = [y_test, y_valid]
        new_y_cols = []

        feature_matrix["target"] = categorical_convert.fit_transform(feature_matrix["target"])

        for y in y_cols:
            all_values = np.unique(list(y.values))
            diff = np.setdiff1d(all_values, categorical_convert.classes_)
            categorical_convert.classes_ = np.concatenate((categorical_convert.classes_, diff))
            y = categorical_convert.transform(y)
            new_y_cols.append(y)

        return new_y_cols[0], new_y_cols[1], feature_matrix

    def get_preds_multi(self, y_hat, y_test):

        pred_data = pd.DataFrame(y_hat)
        pred_data['max'] = pred_data.idxmax(axis=1)
        pred_data["target"] = np.zeros((len(y_test),))
        for i in range(len(y_test)):
            pred_data.loc[i, "target"] = y_test.iloc[i]

        pred_data["target"] = pred_data["target"].astype(int)
        return pred_data

    def precision_function(self, pred_data):

        AP = []
        rec_levels = np.linspace(0, 1, num=11)
        num_classes = 5

        for cls in range(num_classes):
            ranked_prob = np.array(pred_data.loc[:, cls]).argsort()[
                          ::-1]

            max_cls = pred_data.iloc[ranked_prob]["max"]
            target_cls = pred_data.iloc[ranked_prob]["target"]

            perf = pd.DataFrame(columns=["TP", "FP", "FN", "P", "R"])

            for r in range(pred_data.shape[0]):
                if max_cls[r] == cls and target_cls[r] == cls:
                    perf.loc[r, "TP"] = 1
                    perf.loc[r, "FN"] = 0
                    perf.loc[r, "FP"] = 0
                elif max_cls[r] != cls and target_cls[r] == cls:
                    perf.loc[r, "FN"] = 1
                    perf.loc[r, "TP"] = 0
                    perf.loc[r, "FP"] = 0
                elif max_cls[r] == cls and target_cls[r] != cls:
                    perf.loc[r, "FP"] = 1
                    perf.loc[r, "TP"] = 0
                    perf.loc[r, "FN"] = 0
                elif max_cls[r] != cls and target_cls[r] != cls:
                    perf.loc[r, "FP"] = 0
                    perf.loc[r, "TP"] = 0
                    perf.loc[r, "FN"] = 0
                    perf.loc[r, "R"] = 0
                    perf.loc[r, "P"] = 0
                    continue

                TP = (perf.iloc[:r + 1]["TP"]).sum()
                FP = (perf.iloc[:r + 1]["FP"]).sum()
                FN = (perf.iloc[:r + 1]["FN"]).sum()

                if (TP + FP) != 0:
                    perf.loc[r, "P"] = TP / (TP + FP)
                else:
                    perf.loc[r, "P"] = 0

                if (TP + FN) != 0:
                    perf.loc[r, "R"] = TP / (TP + FN)
                else:
                    perf.loc[r, "R"] = 0

            prec_lists = [perf.loc[perf['R'] >= i, 'P'].tolist() for i in rec_levels]
            maxAP = [max(pl) if pl else 0 for pl in prec_lists]

            if maxAP != []:
                meanAP = np.sum(maxAP) / len(rec_levels)
            else:
                meanAP = 0

            AP.append(meanAP)

        mean_ap = np.mean(AP)

        return mean_ap

    def plot_confusion_matrix(self, predictions):

        # predictions = np.load("/data2/hic_lstm/data/graph/predictions.npy")
        conf_matrix = confusion_matrix(predictions[:, 7], predictions[:, 6])
        conf_matrix = conf_matrix[1:, 1:]
        df_cm = pd.DataFrame(conf_matrix)
        df_cm = df_cm.div(df_cm.sum(axis=0), axis=1)

        x_axis_labels = ["A2", "A1", "B1", "B2", "B3"]
        y_axis_labels = ["A2", "A1", "B1", "B2", "B3"]

        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d", xticklabels=x_axis_labels,
                   yticklabels=y_axis_labels)  # font size

        plt.show()

        print("done")

    def get_pos_data(self, window_labels):
        rna_window_labels = window_labels.loc[
            (window_labels["start"] > self.start) & (window_labels["start"] < self.stop)].reset_index()

        rna_window_labels = rna_window_labels.reset_index(drop=True)
        functional_data = self.get_window_data(rna_window_labels)

        return functional_data

    def subc_baseline(self, Subcompartments, window_labels, mode):

        # subc_features = pd.DataFrame(columns=list(np.arange(self.num_subc)))
        sc_ob = Subcompartments(self.cfg, "GM12878", self.chr, mode="Rao")
        sc_data = sc_ob.get_sc_data()

        sc_data = sc_data.filter(['start', 'end', 'target'], axis=1)
        sc_data = sc_data.drop_duplicates(keep='first').reset_index(drop=True)
        sc_data = self.add_cum_pos(sc_data, mode="ends")
        sc_data = sc_data.replace({'target': {-1: 3, -2: 4, -3: 5}, })
        sc_data = sc_data.loc[sc_data["target"] != 0]

        if mode == "ends":
            functional_data = self.get_pos_data(window_labels)
        else:
            functional_data = window_labels

        sc_functional_data = self.get_pos_data(sc_data)
        sc_functional_data = sc_functional_data.rename(columns={"target": "label"})
        sc_functional_data = sc_functional_data.dropna()
        sc_merged_data = pd.merge(sc_functional_data, functional_data, on=['pos'])

        nR = len(sc_merged_data)
        temp = np.zeros((nR, self.num_subc + 1))
        temp[np.arange(nR), sc_merged_data["label"].astype(int)] = 1

        temp = temp[:, 1:]
        temp_df = pd.DataFrame(temp)
        temp_df["target"] = sc_merged_data["target"]

        return temp_df

    def plot_pr_curve(self, precision, recall):

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')
        plt.savefig('XGBoost_PR')
        plt.show()
        pass

    def calculate_map(self, feature_matrix, mode, exp):
        mean_map = 0
        n_folds = 3
        n_recall = 11

        if exp == "map" or exp == "confusion":
            feature_size = self.cfg.pos_embed_size
        elif exp == "baseline":
            feature_size = self.num_subc

        average_precisions = np.zeros(n_folds)
        feature_matrix = feature_matrix.dropna()
        feature_matrix = feature_matrix.replace({'target': {-1: 3, -2: 4, -3: 5}, })
        feature_matrix = feature_matrix.sample(frac=1)
        predictions = pd.DataFrame()

        for i in range(n_folds):
            X_train = pd.DataFrame(columns=list(np.arange(feature_size)))
            y_train = pd.DataFrame()
            X_test = feature_matrix.iloc[i::n_folds, 0:feature_size]
            X_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds, 0:feature_size]
            y_test = feature_matrix.iloc[i::n_folds]["target"]
            y_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds]["target"]

            for j in range(n_folds):
                if j != i and j != (i + 1) % n_folds:
                    fold_mat = feature_matrix.iloc[j::n_folds, 0:feature_size]
                    y_mat = feature_matrix.iloc[j::n_folds]["target"]
                    X_train = pd.concat([X_train, fold_mat])
                    y_train = pd.concat([y_train, y_mat])

            y_train = y_train.astype(int)

            if mode == "multi":
                model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 12), max_depth=6,
                                              objective='multi:softmax', num_class=6)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mlogloss',
                          early_stopping_rounds=20,
                          verbose=False)
            else:
                model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 12), max_depth=6)

                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20,
                          verbose=False)

            y_hat = model.predict_proba(X_test)
            y_test = y_test.astype(int)

            if mode == "multi":
                pred_data = self.get_preds_multi(y_hat, y_test)
                if exp == "confusion":
                    predictions = pd.concat([predictions, pred_data])
                else:
                    average_precisions[i] = self.precision_function(pred_data)

            elif mode == "binary":
                average_precisions[i] = average_precision_score(y_test, y_hat[:, 1])

            # precision, recall, _ = precision_recall_curve(y_test, y_hat[:, 1])
            # self.plot_pr_curve(precision, recall)

        # np.save('/data2/hic_lstm/downstream/RNA-seq/recall.npy', recall)
        if exp == "confusion":
            self.plot_confusion_matrix(predictions)
        else:
            mean_map = np.nanmean(average_precisions)

        if np.isnan(mean_map):
            mean_map = 0

        return mean_map

    def mlp_regressor(self, features):
        mode = "mlp"

        target_column = ['target']
        pos_column = ['pos']
        predictors = list(set(list(features.columns)) - set(target_column) - set(pos_column))

        X = features[predictors].values
        y = features[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)

        if mode == "mlp":
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32,), activation='relu', solver='adam', max_iter=2000000,
                               learning_rate_init=0.1)
            mlp.fit(X_train, abs(y_train))

            ytrain_predict = mlp.predict(X_train)
            ytest_predict = mlp.predict(X_test)

            r_squared_train = mlp.score(X_train, abs(y_train))
            r_squared_test = mlp.score(X_test, abs(y_test))
        else:
            linear_model = LinearRegression().fit(X_train, abs(y_train))

            y_predict = linear_model.predict(X_test)

            r_squared_test = r2_score(y_true=abs(y_test), y_pred=y_predict,
                                      multioutput="uniform_average")

        return r_squared_test

    def add_cum_pos(self, frame, mode):
        cum_pos = get_cumpos(self.cfg, self.chr)

        if mode == "ends":
            pos_columns = ["start", "end"]
        elif mode == "pos":
            pos_columns = ["pos"]

        frame[pos_columns] += cum_pos

        return frame

    def merge_features_target(self, functional_data):
        feature_matrix = pd.merge(self.embed_rows, functional_data, on="pos")

        feature_matrix = feature_matrix[(feature_matrix[self.feature_columns] != 0).all(axis=1)]
        feature_matrix = feature_matrix.loc[(feature_matrix["target"].isna() != True)]

        return feature_matrix

    def get_feature_matrix(self, rna_window_labels):
        functional_data = self.get_pos_data(rna_window_labels)

        feature_matrix = self.merge_features_target(functional_data)

        return feature_matrix

    def get_zero_pos(self, window_labels, col_list):
        ind_list = []

        # max_len = data.iloc[-1]["y2"]
        max_len = self.start_ends["chr" + str(self.chr)]["stop"]
        mask_vec = np.zeros(max_len, bool)
        n_run = len(col_list) // 2

        for i in range(window_labels.shape[0]):

            count = 0
            for j in range(n_run):
                start = window_labels.loc[i, col_list[count]]
                count += 1
                end = window_labels.loc[i, col_list[count]]
                count += 1

                if start >= max_len or end >= max_len:
                    break

                for k in range(start, end + 1):
                    ind_list.append(k)

        ind_ar = np.array(ind_list)
        mask_vec[ind_ar] = True
        zero_vec = np.invert(mask_vec)
        zero_ind = np.nonzero(zero_vec)
        zero_frame = pd.DataFrame(np.transpose(zero_ind), columns=['pos'])
        zero_frame["target"] = pd.Series(np.zeros(len(zero_frame))).astype(int)
        return zero_frame

    def get_window_data(self, frame):
        functional_data = pd.DataFrame(columns=["pos", "target"])
        if frame.index[0] == 1:
            frame.index -= 1

        for i in range(0, frame.shape[0]):

            start = frame.loc[i, "start"]
            end = frame.loc[i, "end"]

            for j in range(start, end + 1):
                functional_data = functional_data.append({'pos': j, 'target': frame.loc[i, "target"]},
                                                         ignore_index=True)

        return functional_data

    def balance_classes(self, feature_matrix):
        if feature_matrix["target"].value_counts().index[0] == 1:
            bal_mode = "undersampling"
        else:
            bal_mode = "oversampling"

        feature_matrix = self.fix_class_imbalance(feature_matrix, mode=bal_mode)

        return feature_matrix

    def fix_class_imbalance(self, feature_matrix, mode='undersampling'):
        if mode == 'undersampling':
            feat_majority = feature_matrix[feature_matrix.target == 1]
            feat_minority = feature_matrix[feature_matrix.target == 0]

            feat_majority_downsampled = resample(feat_majority,
                                                 replace=False,
                                                 n_samples=feat_minority.shape[0],
                                                 random_state=123)

            feature_matrix = pd.concat([feat_majority_downsampled, feat_minority]).reset_index(drop=True)

        elif mode == 'oversampling':
            feat_majority = feature_matrix[feature_matrix.target == 0]
            feat_minority = feature_matrix[feature_matrix.target == 1]

            feat_minority_upsampled = resample(feat_minority,
                                               replace=True,
                                               n_samples=feat_majority.shape[0],
                                               random_state=123)

            feature_matrix = pd.concat([feat_minority_upsampled, feat_majority]).reset_index(drop=True)

        return feature_matrix


if __name__ == '__main__':
    cfg = Config()
    chr = 21
    helper_ob = DownstreamHelper(cfg, chr, mode="lstm")
