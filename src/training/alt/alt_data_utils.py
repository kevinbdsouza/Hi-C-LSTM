import numpy as np
import pandas as pd
import torch
import torch.utils.data
from training.alt.alt_config import Config


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


def convert_indices(input_pairs, cum_pos):
    input_pairs = input_pairs - cum_pos
    input_pairs[input_pairs < 0] = 0
    return input_pairs

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
        data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (cfg.hic_path, cfg.cell, chr, chr), sep="\t",
                           names=['i', 'j', 'v'])
        data = data.dropna()
        data[['i', 'j']] = data[['i', 'j']] / cfg.resolution
        data[['i', 'j']] = data[['i', 'j']].astype('int64')
        return data
    except Exception as e:
        print("Hi-C txt file does not exist or error during Juicer extraction")


def get_hicmat(data, chr, cfg):
    data = data.apply(pd.to_numeric)
    nrows = max(data['i'].max(), data['j'].max())
    seq_diff = cfg.sequence_length_pos - (nrows % cfg.sequence_length_pos)
    nrows_full = nrows + seq_diff

    data['v'] = data['v'].fillna(0)
    rows = np.array(data["i"]).astype(int)
    cols = np.array(data["j"]).astype(int)

    hic_mat = np.zeros((nrows + 1, nrows + 1))
    hic_mat[rows, cols] = np.array(data["v"])
    hic_upper = np.triu(hic_mat)
    hic_mat[cols, rows] = contactProbabilities(np.array(data["v"]))
    hic_lower = np.tril(hic_mat)
    hic_mat = hic_upper + hic_lower
    hic_mat[np.diag_indices_from(hic_mat)] /= 2
    hic_mat[0 , 0] = 0

    indices = np.arange(1, nrows + 1)
    cum_idx = get_bin_idx(np.full(nrows, chr), indices, cfg)
    indices = np.zeros(nrows_full, )
    indices[:nrows] = cum_idx
    return hic_mat, indices, nrows


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

    hic_mat, indices, nrows = get_hicmat(data, chr, cfg)

    values = torch.from_numpy(hic_mat)
    indices = torch.from_numpy(indices)
    return indices, values, nrows


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
    cum_idx, values, nrows = get_samples_sparse(data, chr, cfg)

    return cum_idx, values, nrows


def get_data_loader_chr(cfg, chr, shuffle=True):
    """
    get_data(cfg, chr, shuffle) -> DataLoader
    Uses saved processed input indices and Hi-C values to load single chromosome.
    Creates DataLoader.
    Supports only those chromosomes for which processed data exists.
    Create processed data by calling the save_processed_data function.
    Args:
        cfg (Config): the configuration to use for the experiment.
        chr (int): the chromosome to load Hi-C data for.
        shuffle (bool): To shuffle examples or not
    Raises:
        Error: Processed data does not exist for chromosome.
        """

    try:
        input_idx = torch.load(cfg.processed_data_dir + 'input_idx_chr' + str(chr) + '.pth')
        values = torch.load(cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')

        "create dataloader"
        dataset = torch.utils.data.TensorDataset(input_idx.float(), values.float())
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=shuffle)
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

        cum_idx, val = get_data(cfg, chr)
        torch.save(cum_idx, cfg.processed_data_dir + 'cum_idx_chr' + str(chr) + '.pth')
        torch.save(val, cfg.processed_data_dir + 'values_chr' + str(chr) + '.pth')


if __name__ == "__main__":
    cfg = Config()
    cell = cfg.cell
    save_processed_data(cfg)
