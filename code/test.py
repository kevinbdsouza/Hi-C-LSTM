import torch
import numpy as np

import config
from data_utils import get_data, load_hic, get_bedfile, get_data_loader_chr
from model import SeqLSTM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

cfg = config.Config()
cell = "GM12878"
chr = 21

model = SeqLSTM(cfg, device).to(device)

#get data
input_idx, values, sample_index = get_data(model, cfg, cell, chr)

data_loader, sample_index = get_data_loader_chr(model, cfg, cell, chr, dense=False)

bed_file = get_bedfile(np.unique(sample_index[:,:2], axis=0), cfg)
embeddings = model.get_embeddings(np.unique(sample_index[:,:2], axis=0))

    #data = load_hic(cfg, cell, chr)