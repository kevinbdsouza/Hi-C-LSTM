import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import training.config as config
#import matplotlib as mpl
#mpl.use('module://backend_interagg')

#matplotlib.use('pdf')


def get_cumpos(cfg, chr_num):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    if chr_num == 1:
        cum_pos = 0
    else:
        key = "chr" + str(chr_num - 1)
        cum_pos = sizes[key]

    return cum_pos


def get_bin_idx(chr, pos, cfg):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] // 50 for key in chr]

    return pos + chr_start


def get_genomic_coord(chr, bin_idx, cfg):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] for key in chr]

    return (bin_idx - chr_start) * cfg.resolution


def load_hic(cfg, cell, chr):
    data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (cfg.hic_path, cell, chr, chr), sep="\t", names=['i', 'j', 'v'])

    data = data.dropna()
    data[['i', 'j']] = data[['i', 'j']] / cfg.resolution
    data[['i', 'j']] = data[['i', 'j']].astype('int64')
    return data


def get_samples_dense(data, chr, cfg):
    data = data.apply(pd.to_numeric)
    data['v'] = data['v'].fillna(0)
    dist = cfg.distance_cut_off_mb
    mat = scipy.sparse.coo_matrix((data.v, (data.i, data.j))).tocsr()

    # bin1 = get_bin_idx(chr, 0, cfg)
    nrows = mat.shape[0]

    values = torch.zeros(nrows, 2 * dist)
    input_idx = torch.zeros(nrows, 2 * dist, 2)

    for row in range(nrows):
        # get distance around diagonal
        start = max(row - dist, 0)
        stop = min(row + dist, mat.shape[0])

        # get Hi-C values
        vals = mat[row, start:stop].todense()
        vals = contactProbabilities(vals)
        vals = torch.squeeze(torch.from_numpy(vals))

        # get indices for inserting data
        idx1 = max(0, dist - row)
        idx2 = idx1 + vals.shape[0]

        # insert values
        # vals_tmp = torch.zeros(1, 2 * dist)
        values[row, idx1:idx2] = vals

        # get indices
        j = torch.tensor(np.arange(start, stop)).unsqueeze(dim=1)
        i = torch.full(j.shape, fill_value=row)
        ind = torch.cat((i, j), 1)
        input_idx[row, idx1:idx2, ] = ind

    # only add datapoint if one of the values is non-zero:
    nonzero_idx = torch.sum(values, dim=1).nonzero().squeeze()
    values = values[nonzero_idx,]
    input_idx = input_idx[nonzero_idx,]
    sample_index = None

    return input_idx, values, sample_index


def get_samples_sparse(data, chr, cfg):
    window_model = False
    data = data.apply(pd.to_numeric)
    nrows = max(data['i'].max(), data['j'].max()) + 1
    data['v'] = data['v'].fillna(0)
    data['i_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['i'], cfg)
    data['j_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['j'], cfg)

    values = []
    frame_end_window = []
    input_idx = []
    nvals_list = []
    sample_index = []
    for row in range(nrows):
        # get Hi-C values
        vals = data[data['i'] == row]['v'].values
        nvals = vals.shape[0]
        if nvals == 0:
            continue
        else:
            vals = contactProbabilities(vals)

        if (nvals > 10):
            nvals_list.append(nvals)
            vals = torch.from_numpy(vals)

            split_vals = list(vals.split(cfg.sequence_length, dim=0))

            # get indices
            j = torch.Tensor(data[data['i'] == row]['j_binidx'].values)
            i = torch.Tensor(data[data['i'] == row]['i_binidx'].values)

            # concatenate indices
            ind = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1)), 1)
            split_ind = list(torch.split(ind, cfg.sequence_length, dim=0))

            if window_model:
                dist = cfg.distance_cut_off_mb
                for i in range(len(split_ind) - 1):
                    win_ind = torch.cat((split_ind[i][-dist:, :], split_ind[i + 1][-dist:, :]),
                                        0)
                    win_vals = torch.cat((split_vals[i][-dist:], split_vals[i + 1][-dist:]),
                                         0)
                    split_ind.append(win_ind)
                    split_vals.append(win_vals)

            input_idx = input_idx + split_ind
            values = values + split_vals
            sample_index.append(ind)

    values = pad_sequence(values, batch_first=True)
    input_idx = pad_sequence(input_idx, batch_first=True)

    sample_index = np.vstack(sample_index)
    sample_index = np.concatenate((np.full((sample_index.shape[0], 1), chr), sample_index), 1).astype('int')

    return input_idx, values, sample_index


def window_model(input_idx_list, values_list, input_idx, values):
    window = 10
    num_win_list = []
    num_last_list = []
    for chr_id in range(21):
        max_len = len(input_idx_list[0])
        num_win_list[chr_id] = int(max_len / window)
        num_last_list[chr_id] = int(max_len % window)

    max_num_win = int(max(num_win_list)) + 1
    for l in range(max_num_win):
        for chr_id in range(21):
            val_chr = values_list[chr_id]
            idx_chr = input_idx_list[chr_id]

            num_win = num_win_list[chr_id]
            num_last = num_last_list[chr_id]
            if l == num_win:
                val_chr_chunk = val_chr[num_win * window:(num_win * window) + num_last]
                idx_chr_chunk = idx_chr[num_win * window:(num_win * window) + num_last, :, :]
            else:
                val_chr_chunk = val_chr[num_win * window:(num_win + 1) * window, :]
                idx_chr_chunk = idx_chr[num_win * window:(num_win + 1) * window, :, :]

            try:
                values = torch.cat((values, val_chr_chunk), 0)
                input_idx = torch.cat((input_idx, idx_chr_chunk), 0)
            except Exception as e:
                continue

    return input_idx, values


def contactProbabilities(values, delta=1e-10):
    coeff = np.nan_to_num(1 / (values + delta))
    CP = np.power(1 / np.exp(8), coeff)

    return CP


def get_samples(data, chr, cfg, dense):
    if dense:
        return get_samples_dense(data, chr, cfg)
    else:
        return get_samples_sparse(data, chr, cfg)


def get_data(cfg, cell, chr):
    data = load_hic(cfg, cell, chr)
    input_idx, values, sample_index = get_samples(data, chr, cfg, dense=False)

    return input_idx, values, sample_index


def get_data_loader_chr(cfg, chr):
    # input_idx, values, sample_index = get_data(cfg, cell, chr)
    input_idx = torch.load(cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
    values = torch.load(cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')
    sample_index = torch.load(cfg.processed_data_dir + 'sample_index_chr' + str(chr) + '.pth')

    # create dataset, dataloader
    dataset = torch.utils.data.TensorDataset(input_idx.float(), values.float())
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)

    return data_loader, sample_index


def get_data_loader_batch_chr(cfg):
    values = torch.empty(0, cfg.sequence_length)
    input_idx = torch.empty(0, cfg.sequence_length, 2)

    sample_index_list = []

    # chr_list = [2, 22, 10, 12, 7, 3, 16, 11, 20, 4, 19, 15, 18, 8, 14, 6, 17, 21]
    # chr_list = [2, 22, 10, 12, 3, 16, 11, 20, 4, 19, 9, 15, 5, 18, 8, 14, 6, 17, 13, 21, 1, 7]
    #chr_list = [22, 12, 16, 11, 20, 19, 15, 18, 14, 17, 13, 21]
    chr_list = [15, 16, 17, 18, 19, 20, 21]

    for chr in chr_list:
        idx = torch.load(cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
        val = torch.load(cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')
        sample_idx = torch.load(cfg.processed_data_dir + 'sample_index_chr' + str(chr) + '.pth')

        values = torch.cat((values, val.float()), 0)
        input_idx = torch.cat((input_idx, idx), 0)
        sample_index_list.append(sample_idx)

    sample_index_tensor = np.vstack(sample_index_list)
    # create dataset, dataloader
    dataset = torch.utils.data.TensorDataset(input_idx, values)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

    return data_loader, sample_index_tensor


def save_processed_data(cfg, cell):
    # chr_list = [15, 16, 17, 18, 19, 20, 21]
    chr_list = [2, 22, 10, 12, 3, 16, 11, 20, 4, 19, 9, 15, 5, 18, 8, 14, 6, 17, 13, 21, 1, 7]
    for chr in chr_list:
        print(chr)
        idx, val, sample_idx = get_data(cfg, cell, chr)

        torch.save(idx, cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
        torch.save(val, cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')
        torch.save(sample_idx, cfg.processed_data_dir + 'sample_index_chr' + str(chr) + '.pth')

    pass


def get_data_loader(cfg, cell):
    # if(len(os.listdir(cfg.processed_data_dir)) != 0):
    #    #load input data
    #    input_idx = cfg.processed_data_dir + 'input_idx.pth'
    #    values = cfg.processed_data_dir + 'values.pth'
    #    sample_index = cfg.processed_data_dir + 'input_index.pth'

    # else:
    # create input dataset
    values = torch.empty(0, cfg.sequence_length)
    input_idx = torch.empty(0, cfg.sequence_length, 2)
    sample_index = []

    # for chr in list(range(1, 11)) + list(range(12, 23)):
    for chr in list(range(1, 23)):
        idx, val, sample_idx = get_data(cfg, cell, chr)

        values = torch.cat((values, val.float()), 0)
        input_idx = torch.cat((input_idx, idx), 0)
        sample_index.append(sample_idx)

    sample_index = np.vstack(sample_index)

    # save input data
    torch.save(input_idx, cfg.processed_data_dir + 'input_idx.pth')
    torch.save(values, cfg.processed_data_dir + 'values.pth')
    torch.save(sample_index, cfg.processed_data_dir + 'input_index.pth')

    # create dataset, dataloader
    dataset = torch.utils.data.TensorDataset(input_idx, values)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)

    return data_loader, sample_index


def get_bedfile(sample_index, cfg):
    chr = sample_index[:, 0]
    bin_idx = sample_index[:, 1]
    start_coord = get_genomic_coord(chr, bin_idx, cfg)
    stop_coord = start_coord + cfg.resolution

    bedfile = pd.DataFrame({'chr': chr, 'start': start_coord, 'stop': stop_coord})
    return bedfile

def scHiC(cfg, cell):
    file_name = "/GSM2254215_ML1.validPairs.txt"
    full_path = cfg.hic_path + cell + file_name
    pairs = pd.read_csv(full_path, sep="\t", names=['chrA', 'x1', 'x2', 'chrB', 'y1', 'y2', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'])

    pairs_19 = pairs.loc[pairs["chrA"] == "human_chr19"]
    pairs_20 = pairs.loc[pairs["chrA"] == "human_chr20"]
    pairs_21 = pairs.loc[pairs["chrA"] == "human_chr21"]
    pairs = pairs.loc[pairs["chrA"] == "human_chr22"]
    

    print("done")

    pass

if __name__ == "__main__":
    cfg = config.Config()
    cell = cfg.cell
    #save_processed_data(cfg, cell)
    scHiC(cfg, cell)
