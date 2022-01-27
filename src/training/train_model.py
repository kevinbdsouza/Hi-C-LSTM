import numpy as np
import torch
import os
import sys
import time

from torch.utils.tensorboard import SummaryWriter

import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader, get_bedfile, get_data_loader_batch_chr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(cfg, model_name, cell, writer):
    # initalize model
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    optimizer, criterion = model.compile_optimizer(cfg)

    # get data
    start = time.time()
    data_loader, samples = get_data_loader_batch_chr(cfg)
    print("%s batches loaded" % str(len(data_loader)))
    end = time.time()
    print("Time to obtain data: %s" % str(end - start))

    # train model
    model.train_model(data_loader, criterion, optimizer, writer)

    # save model
    torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')

    # get model embeddings with bed file of genomic coordinates
    # bed_file = get_bedfile(np.unique(samples[:, :2], axis=0), cfg)
    # embeddings = model.get_embeddings(np.unique(samples[:, 2], axis=0))

    # bed_file.to_csv(cfg.output_directory + "combined150_embedding_coordinates_chr17.bed", sep='\t', header=False, index=False)
    # np.save(cfg.output_directory + "combined_150_cp_embeddings_chr21.npy", embeddings)


if __name__ == '__main__':
    cfg = config.Config()
    cell = cfg.cell
    # model_name = sys.argv[1]
    #model_name = "shuffle_" + cell
    model_name = "shuffle_GM12878_test"

    # set up tensorboard logging
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('./tensorboard_logs/' + model_name + timestr)

    train_model(cfg, model_name, cell, writer)

    print("done")
