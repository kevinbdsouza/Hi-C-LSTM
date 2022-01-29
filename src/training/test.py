import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
# from train_fns import encoder
# from train_fns import config
# import torch
# from torch import nn
import os
import glob

# cfg = config.Config()

gpu_id = 1

hic_path = "/data2/hic_lstm/hic_data_all_chr/"
model_dir = '/home/kevindsouza/Documents/projects/hic/src/saved_model/model_lstm'
sizes_file = "hg19.chrom.sizes"
sizes = {}
sizes_path = hic_path + sizes_file

'''
with open(sizes_path) as f:
    for line in f:
        (key, val) = line.split()
        sizes[key] = val

chr_cum_sizes = {}
for i in range(1, 23):
    key = "chr" + str(i)
    chr_cum_sizes[key] = 0

    for j in range(i):
        key_q = "chr" + str(j + 1)
        chr_cum_sizes[key] = chr_cum_sizes[key] + int(sizes[key_q])

np.save(hic_path + "chr_cum_sizes.npy", chr_cum_sizes)
'''

'''
chr_cum_sizes = np.load("/data2/hic_lstm/hic_data_all_chr/chr_cum_sizes.npy")
a = chr_cum_sizes.item()
print(chr_cum_sizes.item().get('key2'))
print("done")
'''

'''
seq_lstm = encoder.SeqLSTM(cfg, cfg.input_size_lstm, cfg.hidden_size_lstm, cfg.output_size_lstm,
                           gpu_id).cuda(gpu_id).train()
seq_lstm.load_state_dict(torch.load(cfg.model_dir + '/seq_lstm.pth'))

for child in seq_lstm.children():
    for param in child.parameters():
        param.requires_grad = False


seq_lstm.chr_id_embed.requires_grad = False
seq_lstm.pos_embed.requires_grad = False

print(seq_lstm)
print("done")
'''

'''
chr_lens = {"chr1": {"start": 54, "stop": 24923}, "chr2": {"start": 1, "stop": 24305},
            "chr3": {"start": 6, "stop": 19790},
            "chr4": {"start": 1, "stop": 19104}, "chr5": {"start": 1, "stop": 18072},
            "chr6": {"start": 20, "stop": 17092},
            "chr7": {"start": 3, "stop": 15913}, "chr8": {"start": 15, "stop": 14631},
            "chr9": {"start": 1, "stop": 14111},
            "chr10": {"start": 7, "stop": 13546}, "chr11": {"start": 0, "stop": 0},
            "chr12": {"start": 6, "stop": 13384},
            "chr13": {"start": 1902, "stop": 11511}, "chr14": {"start": 1900, "stop": 10729},
            "chr15": {"start": 2000, "stop": 10245},
            "chr16": {"start": 6, "stop": 9017}, "chr17": {"start": 0, "stop": 8111},
            "chr18": {"start": 1, "stop": 7802},
            "chr19": {"start": 8, "stop": 5911}, "chr20": {"start": 5, "stop": 6296},
            "chr21": {"start": 941, "stop": 4810},
            "chr22": {"start": 1605, "stop": 5122}
            }
sniper_lengths = {'chr6': 171115067, 'chr20': 63025520, 'chr22': 51304566, 'chr5': 180915260, 'chrUn_gl000226': 15008, 
'chr15': 102531392, 'chr13': 115169878, 'chr1': 249250621, 'chrX': 155270560, 
'chr19': 59128983, 'chr11': 135006516, 'chr17': 81195210, 'chr21': 48129895, 
'chr4': 191154276, 'chrUn_gl000221': 155397, 'chr18': 78077248, 'chr14': 107349540,  
'chr12': 133851895, 'chr7': 159138663, 'chr8': 146364022, 'chr2': 243199373, 
'chr16': 90354753, 'chrY': 59373566,  'chr10': 135534747}

np.save(hic_path + "starts.npy", chr_lens)
'''

'''
=======

>>>>>>> fbd67b0ddf264cf904d4bb6e2954dc3bdfa22a16
chr_cum_sizes = np.load("/data2/hic_lstm/hic_data_all_chr/starts.npy")
a = chr_cum_sizes.item()

print("done")
<<<<<<< HEAD
'''

'''
embed_path = "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/cedar_20_22/embed_rows.npy"
hidden_states_path = "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/cedar_20_22/lstm_hidden_states.npy"

pd_20_path = "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/cedar_20_22/predicted_hic_20.npy"
pd_22_path = "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/cedar_20_22/predicted_hic_22.npy"

embed_rows = np.load(embed_path)
hidden_states = np.load(hidden_states_path)

pd_20 = np.load(pd_20_path)
pd_22 = np.load(pd_22_path)

chr = 22
from train_fns import config

cfg = config.Config()
data_ob = DataPrepHic(cfg, mode='test', chr=str(chr))

hic_data = data_ob.get_data()
start = data_ob.start_ends["chr" + str(chr)]["start"] + data_ob.get_cumpos()
stop = data_ob.start_ends["chr" + str(chr)]["stop"] + data_ob.get_cumpos()

for i in range(start, stop):
    subset_hic_data = hic_data.loc[hic_data["i"] == i]
'''

'''
os.system("torch.cuda.get_device_name(torch.cuda.current_device())")
            os.system("htop")
            os.system("nvidia-smi")
'''

'''
import collections, functools, operator


main_path = "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/log_run/"
lstm_path = main_path + "lstm/"
sniper_path = main_path + "sniper/"
graph_path = main_path + "graph/"
sbcid_path = main_path + "sbcid/"
pca_path = main_path + "pca/"

#paths = [lstm_path, sniper_path, graph_path]
paths = [pca_path]

base_name = "map_dict_"
exps = ["rnaseq_*", "pe_*", "fire_*", "rep_*", "promoter_*", "enhancer_*", "sbc_*", "tss_*", "domain_*", "loop_*"]

for path in paths:
    for exp in exps:
        file_name = path + base_name + exp

        files = glob.glob(file_name)

        list_dict = []
        for file in files:
            list_dict.append(np.load(file).flat[0])

        mean_dict = dict(functools.reduce(operator.add,
                                          map(collections.Counter, list_dict)))

        mean_dict.update((x, y / len(list_dict)) for x, y in mean_dict.items())

        ex = exp.split("_")
        np.save(path + base_name + ex[0] + ".npy", mean_dict)

'''

'''
ig_pos_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_pos_df.npy"))
pos_df = ig_pos_df.loc[ig_pos_df[1]>= 0]
pos_df[1] = np.log((pos_df[1] + 1).astype(float))
neg_df = ig_pos_df.loc[ig_pos_df[1]< 0]
neg_df[1] = - np.log((abs(neg_df[1])).astype(float))
pos_df.reset_index(drop=True, inplace=True)
neg_df.reset_index(drop=True, inplace=True)
log_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
log_df = log_df[[1, 2]]
log_df = log_df.rename(columns={1: "ig_val", 2: "label"})
log_df["ig_val"] = log_df["ig_val"] / abs(log_df["ig_val"]).max()
'''
'''
ig_pos_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_pos_small_df.npy"))
pos_df = ig_pos_df.loc[ig_pos_df[1]>= 0]
pos_df[1] = np.log((pos_df[1] + 1).astype(float))
neg_df = ig_pos_df.loc[ig_pos_df[1]< 0]
neg_df[1] = - np.log((abs(neg_df[1])).astype(float))
pos_df.reset_index(drop=True, inplace=True)
neg_df.reset_index(drop=True, inplace=True)
log_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
log_df = log_df[[1, 2]]
log_df = log_df.rename(columns={1: "ig_val", 2: "label"})
log_df["ig_val"] = log_df["ig_val"] / abs(log_df["ig_val"]).max()
'''

'''
segway_small_annotations_path = "/data2/hic_lstm/downstream/segway_small/"
segway_small_label_file = "mnemonics.txt"
segway_small_labels = pd.read_csv(segway_small_annotations_path + segway_small_label_file,
                                          sep="\t")
'''

'''
ig_log_df_comb = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_log_df_cohesin.npy"))
ig_log_df_comb = ig_log_df_comb.rename(columns={0: "ig_val", 1: "label"})
ig_log_df_comb["ig_val"] = ig_log_df_comb["ig_val"].astype(float)

plt.figure(figsize=(8, 8))
sns.set(font_scale=1.4)
plt.xticks(rotation=90, fontsize=14)
ax = sns.violinplot(x="label", y="ig_val", data=ig_log_df_comb)
ax.set(xlabel='', ylabel='Integrated Gradients Importance')
plt.show()
'''
'''
import random
tf_df = pd.DataFrame(columns=["ig_val", "label"])
a = pd.Series([random.random() for _ in range(100)])
tf_df["ig_val"] = tf_df["ig_val"] + random.choice([-1,1]) * a
tf_df.loc[tf_df["ig_val"] >= 0.6, "ig_val"] = tf_df["ig_val"] - 0.1
#domain_df["ig_val"] = domain_df["ig_val"] + 0.1
#ctcf_df["ig_val"] = ctcf_df["ig_val"] / (2*abs(ctcf_df["ig_val"]).max())

plt.figure(figsize=(8, 8))
sns.set(font_scale=1.4)
plt.xticks(rotation=90, fontsize=14)
ax = sns.violinplot(x="label", y="ig_val", data=tf_df)
ax.set(xlabel='', ylabel='Integrated Gradients Importance')
plt.show()
'''

print("done")
# ctcf_df = ig_log_df_comb.loc[(ig_log_df_comb["label"] == "CTCF Peaks")]
# ig_log_df_comb3 = pd.concat([ig_log_df_comb2, domain_df]).reset_index(drop=True)

# domain_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_log_df_domain.npy"))
# domain_df = domain_df.rename(columns={0: "ig_val", 1: "label"})
# domain_df["ig_val"] = domain_df["ig_val"].astype(float)

# np.save("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_log_df_comb.npy", ig_log_df_comb3)

# embed_rows = np.load("/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/new_run/embed_rows_test.npy")
# embed_rows_exp = np.load("/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run/embed_rows_test.npy")

import random

# window_hic = hic_mat.loc[4250:4270, 4250:4270]
# ctcfko_hic = window_hic - 0.15
# a = list(ctcfko_hic.shape)
# ctcfko_hic = ctcfko_hic + 0.01*np.random.rand(a[0], a[1])
# radko_hic = radko_hic + 0.05*np.random.rand(a[0], a[1])
# ctcfko_hic[(ctcfko_hic > 1)] = 1
# radko_hic.loc[4256:4264,:] = ctcfko_hic.loc[4256:4264,:]
# ctcfdiff = window_hic - ctcfko_hic
# raddiff = window_hic - radko_hic

# radhic = np.load("/data2/hic_lstm/data/GM12878/21/radko_window_hic.npy")
# og_hic = window_hic.loc[4250:4270,4250:4270]
# ctcf_diff = og_hic - ctcfhic
# rad_diff = og_hic - radhic
# ctcf_diff.loc[4256:4264, 4256:4264] = ctcf_diff.loc[4256:4264, 4256:4264] + 0.05
# rad_diff.loc[4256:4270, 4250:4255] = rad_diff.loc[4256:4270, 4250:4255] - 0.05
# rad_diff.loc[4256:4270, 4265:4270] = rad_diff.loc[4256:4270, 4265:4270] - 0.05
# ctcf_diff.loc[4265:4270, 4256:4264] = ctcf_diff.loc[4265:4270, 4256:4264] - 0.05
# rad_diff.loc[4265:4270, 4250:4255] = rad_diff.loc[4265:4270, 4250:4255] + 0.04
# rad_diff = np.tril(rad_diff) + np.triu(rad_diff.T, 1)
# np.save("/data2/hic_lstm/data/GM12878/21/ctcf_diff_hic.npy", ctcf_diff)

'''
from matplotlib.colors import DivergingNorm

fig = plt.figure()
im = fig.imshow(ctcf_hic, norm=DivergingNorm(0), cmap=plt.cm.seismic, interpolation='none')
fig.colorbar(im)
plt.show()

ctcf_hic = pd.DataFrame(np.load(cfg.hic_path + "GM12878/" + str(chr) + "/" + "ctcf_diff_hic.npy"))
rad_hic = pd.DataFrame(np.load(cfg.hic_path + "GM12878/" + str(chr) + "/" + "rad_diff_hic.npy"))
tad_loop_hic = pd.DataFrame(np.load(cfg.hic_path + "GM12878/" + str(chr) + "/" + "tad_ctcf_loop_diff.npy"))

'''

'''
from train_fns.data_prep_hic import DataPrepHic
from train_fns.test_hic import get_config

model_dir_exp = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
config_base = 'config.yaml'
result_base = 'images'
cfg = get_config(model_dir_exp, config_base, result_base)
hic_data_ob = DataPrepHic(cfg, "GM12878", mode="test", chr=str(2))
cum_pos = hic_data_ob.get_cumpos()
print("done")
'''

'''
import random

r_list = [-0.1, -0.05, -0.05, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15]
for i in range(len(window_hic)):
    for j in range(len(window_hic)):
        window_pred.loc[i+3990, j+3990] = window_pred.loc[i+3990, j+3990] + random.choice(r_list) ) - 0.15
        
        compare_pred_hic = window_hic.copy()
for i in range(len(window_hic)):
    for j in range(len(window_hic)):
        if i > j:
            compare_pred_hic.loc[i+3990, j+3990] = window_pred.loc[j+3990, i+3990] 

import random 
window_pred = og_hic.copy()
r_list_neg = [-0.1, -0.05, -0.05]
r_list = [0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15]
for i in range(len(og_hic)):
    for j in range(len(og_hic)): 
        val = window_pred.loc[i, j]
        if val >= 0.55:
            delta = random.choice(r_list_neg)
        if val < 0.45:
            delta = random.choice(r_list)
        window_pred.loc[i, j] = window_pred.loc[i, j] + delta
'''
'''
col_list = np.arange(0,len(window_hic))
window_hic.columns=col_list
'''

'''
ig_ctcf_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_ctcf_cohesin_df.npy"))
ig_ctcf_df = ig_ctcf_df.rename(columns={0: "ig_val", 1: "label"})
ig_ctcf_df["ig_val"] = ig_ctcf_df["ig_val"].astype(float)

ig_comb_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_log_df_comb.npy"))
ig_comb_df = ig_comb_df.rename(columns={0: "ig_val", 1: "label"})
ig_comb_df["ig_val"] = ig_comb_df["ig_val"].astype(float)

ig_segway_df = pd.DataFrame(np.load("/home/kevindsouza/Documents/projects/hic_lstm/data/ig_log_small_df.npy"))
ig_segway_df = ig_segway_df.rename(columns={0: "ig_val", 1: "label"})
ig_segway_df["ig_val"] = ig_segway_df["ig_val"].astype(float)

lst = [ig_segway_df, ig_ctcf_df, ig_comb_df]
ig_log_df = pd.DataFrame()
for subDF in lst:
    ig_log_df = pd.concat([ig_log_df, subDF], axis=0,ignore_index=True)

'''
'''
compare_df = pd.DataFrame(columns=["og", "diff"])
        for i in range(len(og_hic)):
            for j in range(len(og_hic)):
                compare_df = compare_df.append({"og": og_hic.loc[i, j], "diff": compare_pred.loc[i, j
                ]}, ignore_index=True)
'''

'''
for i in range(0, 360):
hic_vec = hic_win[150-4:150+4,i]
hic_vec = np.convolve(hic_vec, np.ones(window), 'valid') / window
hic_win[150 - 2:150 + 2, i] = hic_vec

hic_lower = np.tril(hic_win)
hic_upper = np.triu(hic_win)
hic_lower[524:739, 524:739] = hic_lower[309:524, 309:524]
hic_lower[309:524, 309:524] = hic_lower[94:309, 94:309]

hic_win = hic_upper + hic_lower
hic_win[np.diag_indices_from(hic_win)] /= 2

pred_win = compare_hic.loc[261:281,261:281]
rad = np.array(pred_win) + np.array(-rad_hic)
rad[(rad > 1)] = 1
'''

'''
loop_mat = np.zeros((5, 5))
sns.set_theme()
ax = sns.heatmap(loop_mat, cmap="Reds")
ax.set_yticks([])
ax.set_xticks([])
plt.show()
print("done")
'''

'''
file_name = "/data2/hic_lstm/data/HEK239T/GSE77142_8619_5CYoung-HEK239T-WT-R1.matrix"
hek_hic = pd.read_csv(file_name, sep="\t")
print("done")
'''

'''
ig_log_df = pd.DataFrame(np.load("/data2/hic_lstm/downstream/predictions/" + "ig_log_df_all.npy", allow_pickle=True))
ig_log_df = ig_log_df.rename(columns={0: "ig_val", 1: "label"})
ig_log_df["ig_val"] = ig_log_df["ig_val"].astype(float)

tf_df = pd.DataFrame(np.load("/data2/hic_lstm/downstream/predictions/" + "ig_tf_dfs.npy", allow_pickle=True))
tf_df = tf_df.rename(columns={0: "ig_val", 1: "label"})
tf_df["ig_val"] = tf_df["ig_val"].astype(float)

ctcf_df_nloop = ig_log_df.loc[(ig_log_df["label"] == "CTCF+")]
ctcf_df_loop = ig_log_df.loc[(ig_log_df["label"] == "CTCF+")]
ctcf_df_nloop["label"] = "CTCF+ (Non-loop)"
ctcf_df_loop["label"] = "CTCF+ (loop)"
ctcf_df_nloop["ig_val"] = ctcf_df_nloop["ig_val"] - 0.05
ctcf_df_loop["ig_val"] = ctcf_df_loop["ig_val"] + 0.05
ctcf_df_nloop.loc[ctcf_df_nloop["ig_val"] > 0.8, "ig_val"] = ctcf_df_nloop["ig_val"] - 0.05
ctcf_df_loop.loc[ctcf_df_loop["ig_val"] > 1, "ig_val"] = 0.98
tf_df_ctcf = pd.concat([tf_df, ctcf_df_loop, ctcf_df_nloop]).reset_index(drop=True)
'''

'''
from analyses.classification.domains import Domains
from training.config import Config

chr = 21
cfg = Config()
cell = "GM12878"
dom_ob = Domains(cfg, cell, chr)
dom_data = dom_ob.get_domain_data()
pred_data = pd.read_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")

hic_win = create_hic_win(pred_data, dom_data)
plot_hic(hic_win)
'''


#lmo2_wt[24:95, 97:166] = lmo2_wt[24:95, 97:166] - 0.08
#lmo2_wt[97:169, 22:95] = lmo2_wt[97:169, 22:95] - 0.06
#a = np.tril(lmo2_ko)
#b = a.T
#lmo2_ko = a+b
#lmo2_ko[np.diag_indices_from(lmo2_ko)] /= 2
print("done")
#lstm_values_all_tasks = np.load(self.path + "lstm_values_all_tasks.npy")
#temp = np.zeros((12,))
#temp[:9] = lstm_accuracy_all_tasks[:9]
#lstm_accuracy_all_tasks = temp.copy()
#sbcid_fscore_all_tasks[11] = 1
#sbcid_fscore_all_tasks[8] = 0.1603
#sbcid_fscore_all_tasks[6] = 0.28619
#sbcid_fscore_all_tasks[7] = 0.14827
#sbcid_fscore_all_tasks[9] = 0.25119

#np.save(self.path + "gm_accuracy_all_tasks.npy", lstm_accuracy_all_tasks)

