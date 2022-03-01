import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import training.config as config


def get_cumpos(cfg, chr_num):
    """
    get_cumpos(cfg, chr_num) -> int
    Returns cumulative index upto the end of the previous chromosome.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr_num (int): the current chromosome.
    """
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    if chr_num == 1:
        cum_pos = 0
    else:
        key = "chr" + str(chr_num - 1)
        cum_pos = sizes[key]

    return cum_pos


def get_bin_idx(chr, pos, cfg):
    """
    get_bin_idx(chr, pos, cfg) -> List
    Converts genomic coordinates with respect to the given chromosome to cumulative indices.
    Args:
        chr (iterable): the chromosome the bin belongs to.
        pos (List): List of positions to be converted.
        cfg (Config): the configuration to use for the experiment.
    """
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] for key in chr]

    return pos + chr_start


def get_genomic_coord(chr, bin_idx, cfg):
    """
    get_genomic_coord(chr, bin_idx, cfg) -> List
    Converts cumulative indices to genomic coordinates with respect to the given chromosome.
    Args:
        chr (iterable): the chromosome the bin belongs to.
        bin_idx (List): List of bin ids to be converted.
        cfg (Config): the configuration to use for the experiment.
    """
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] for key in chr]

    return (bin_idx - chr_start) * cfg.resolution


def load_hic(cfg, chr):
    """
    load_hic(cfg, chr) -> Dataframe
    Loads data from Hi-C txt files, converts indices to specified resolution.
    Supports only those values of resolution that Juicer can extract from Hi-C txt file.
    Supports only those cell types for which Hi-C txt files exist.
    To check how to create the Hi-C txt file, refer to the documentation.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr (int): the chromosome to extract Hi-C from.
    Raises:
        Error: Hi-C txt file does not exist or error during Juicer extraction.
        Skips: if error during extraction using Juicer Tools, prints out empty txt file
    """
    try:
        data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (cfg.hic_path, cfg.cell, chr, chr), sep="\t", names=['i', 'j', 'v'])
        data = data.dropna()
        data[['i', 'j']] = data[['i', 'j']] / cfg.resolution
        data[['i', 'j']] = data[['i', 'j']].astype('int64')
        return data
    except Exception as e:
        print("Hi-C txt file does not exist or error during Juicer extraction")


def get_samples_sparse(data, chr, cfg):
    """
    get_samples_sparse(data, chr, cfg) -> List, List
    Organizes data into input ids and values.
    Supports varying values of sequence length in the configuration file.
    Args:
        data (DataFrame): the extracted Hi-C data.
        chr (int): the chromosome to extract Hi-C from.
        cfg (Config): the configuration to use for the experiment.
    """
    data = data.apply(pd.to_numeric)
    nrows = max(data['i'].max(), data['j'].max()) + 1
    data['v'] = data['v'].fillna(0)
    data['i_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['i'], cfg)
    data['j_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['j'], cfg)

    values = []
    input_idx = []
    nvals_list = []
    for row in range(nrows):
        vals = data[data['i'] == row]['v'].values
        nvals = vals.shape[0]
        if nvals == 0:
            continue
        else:
            vals = contactProbabilities(vals, smoothing=cfg.hic_smoothing)

        if (nvals > 10):
            nvals_list.append(nvals)
            vals = torch.from_numpy(vals)

            split_vals = list(vals.split(cfg.sequence_length, dim=0))

            "get indices"
            j = torch.Tensor(data[data['i'] == row]['j_binidx'].values)
            i = torch.Tensor(data[data['i'] == row]['i_binidx'].values)

            "conactenate indices"
            ind = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1)), 1)
            split_ind = list(torch.split(ind, cfg.sequence_length, dim=0))

            if cfg.window_model:
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

    "pad sequences if shorter than sequence_length"
    values = pad_sequence(values, batch_first=True)
    input_idx = pad_sequence(input_idx, batch_first=True)

    return input_idx, values


def window_model(input_idx_list, values_list, input_idx, values):
    """
    This function is not fully tested yet. Use only if exploring.
    """
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


def contactProbabilities(values, smoothing=8, delta=1e-10):
    """
    contactProbabilities(values, delta) -> Array
    Squishes Hi-C values between 0 and 1.
    Args:
        values (Array): the Hi-C values.
        smoothing (int): integer coefficent used to exponentially smoothen Hi-C
        delta (float): small positive value to avoid divide by 0.
    """
    coeff = np.nan_to_num(1 / (values + delta))
    contact_prob = np.power(1 / np.exp(smoothing), coeff)

    return contact_prob


def get_data(cfg, chr):
    """
    get_data(cfg, chr) -> List, List
    Loads data from Hi-C txt files, organizes them into input ids and values.
    Supports varying values of sequence length in the configuration file.
    Supports only those values of resolution that Juicer can extract from Hi-C txt file.
    Supports only those cell types for which Hi-C txt files exist.
    To check how to create the Hi-C txt file, refer to the documentation.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr (int): the chromosome to extract Hi-C from.
    Raises:
        Error: Hi-C txt file does not exist or error during Juicer extraction.
        Skips: if error during extraction using Juicer Tools, prints out empty txt file
    """
    data = load_hic(cfg, chr)
    input_idx, values = get_samples_sparse(data, chr, cfg)

    return input_idx, values


def get_data_loader_chr(cfg, chr):
    """
    get_data(cfg, chr) -> DataLoader
    Uses saved processed input indices and Hi-C values to load single chromosome.
    Creates DataLoader.
    Supports only those chromosomes for which processed data exists.
    Create processed data by calling the save_processed_data function.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr (int): the chromosome to load Hi-C data for.
    Raises:
        Error: Processed data does not exist for chromosome.
        """

    try:
        input_idx = torch.load(cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
        values = torch.load(cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')

        "create dataloader"
        dataset = torch.utils.data.TensorDataset(input_idx.float(), values.float())
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)
        return data_loader
    except Exception as e:
        print("Processed data does not exist for chromosome")


def get_data_loader_batch_chr(cfg):
    """
    get_data(cfg) -> DataLoader
    Uses saved processed input indices and Hi-C values.
    Creates DataLoader using all specified chromosomes.
    Use get_data_loader_chr(cfg, chr) if loading only one chromosome.
    Supports only those chromosomes for which processed data exists.
    Create processed data by calling the save_processed_data function.
    Args:
        cfg (Config): the configuration to use for the experiment.
    Raises:
        Error: Processed data does not exist for chromosome.
        Skips: Skips chromosome if error during data loading
    """

    values = torch.empty(0, cfg.sequence_length)
    input_idx = torch.empty(0, cfg.sequence_length, 2)

    for chr in cfg.chr_train_list:
        try:
            idx = torch.load(cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
            val = torch.load(cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')

            values = torch.cat((values, val.float()), 0)
            input_idx = torch.cat((input_idx, idx), 0)
        except Exception as e:
            print("Processed data does not exist for chromosome")
            continue

    "create dataloader"
    dataset = torch.utils.data.TensorDataset(input_idx, values)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True)

    return data_loader


def save_processed_data(cfg):
    """
    save_processed_data(cfg) -> No return object
    Gets data for Hi-C txt files, organizes them into input ids and values.
    Saves them in the processed directory.
    Supports varying values of sequence length in the configuration file.
    Supports only those values of resolution that Juicer can extract from Hi-C txt file.
    Supports only those cell types for which Hi-C txt files exist.
    To check how to create the Hi-C txt file, refer to the documentation.
    Args:
        cfg (Config): the configuration to use for the experiment.
    Raises:
        Error: Hi-C txt file does not exist or error during Juicer extraction.
        Skips: if error during extraction using Juicer Tools, prints out empty txt file
    """
    for chr in cfg.chr_train_list:
        print("Saving input data for Chr", str(chr), "in the specified processed directory")

        idx, val = get_data(cfg, chr)
        torch.save(idx, cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
        torch.save(val, cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')


def scHiC_preprocess(cfg):
    """
    scHiC_preprocess(cfg) -> No return object
    Gets bar codes and positions from pairs file.
    Matched them with bar codes from percentages file.
    Creates a DataFrame corresponding pairwise hg19 contact values.
    Saves the resulting DataFrame in the ScHiC data directory.
    Args:
        cfg (Config): the configuration to use for the experiment.
        """

    chr_list = [19, 20, 21, 22]
    columns = ['x1', 'y1', 'bar1', 'bar2']
    full_pairs_path = cfg.hic_path + cfg.cell + cfg.schic_pairs_file
    pairs = pd.read_csv(full_pairs_path, sep="\t",
                        names=['chrA', 'x1', 'x2', 'chrB', 'y1', 'y2', 'a', 'b', 'c', 'd', 'e', 'bar1', 'bar2',
                               'l', 'i', 'j', 'k'])

    for chr in chr_list:
        pairs = pairs.loc[pairs["chrA"] == "human_chr" + str(chr)]
        pairs = pairs.loc[pairs["chrB"] == "human_chr" + str(chr)]
        pairs = pairs[columns]
        pairs.to_csv(cfg.hic_path + cfg.cell + '/' + str(chr) + '/' + "pairs_" + str(chr) + '.txt', sep="\t")

    full_read_path = cfg.hic_path + cfg.cell + cfg.schic_reads_file
    reads = pd.read_csv(full_read_path, sep="\t",
                        names=['a', 'b', 'reads_hg19', 'd', 'e', 'f', 'bar1', 'bar2', 'i', 'j', 'k', 'l', 'm', 'n',
                               'o', 'p', 'q'])
    reads = reads[['reads_hg19', 'bar1', 'bar2']]

    for chr in chr_list:
        pairs = pd.read_csv(cfg.hic_path + cfg.cell + '/' + str(chr) + '/' + "pairs_" + str(chr) + '.txt', sep="\t")
        merged_pairs = pairs.merge(reads, on=["bar1", "bar2"])
        merged_pairs = merged_pairs[["x1", "y1", "reads_hg19"]]
        merged_pairs = merged_pairs.rename(columns={"x1": "i", "y1": "j", "reads_hg19": "v"})
        merged_pairs.to_csv(cfg.hic_path + cfg.cell + '/' + str(chr) + '/' + "hic_chr" + str(chr) + '.txt', sep="\t")


if __name__ == "__main__":
    cfg = config.Config()
    cell = cfg.cell
    save_processed_data(cfg)
    scHiC_preprocess(cfg)
