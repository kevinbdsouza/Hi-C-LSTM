import torch
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_model(model, cfg, chr):
    """
    train_model(model, cfg, cell, chr) -> No return object
    Loads loads data, tests the model, and saves the predictions in a csv file.
    Works on one chromosome at a time.
    Args:
        model (SeqLSTM): The model that needs to be tested.
        cfg (Config): The configuration to use for the experiment.
        cell (string): The cell type to extract Hi-C from.
        chr (int): The chromosome to test.
    """

    "get data"
    data_loader = get_data_loader_chr(cfg, chr)

    "test model"
    predictions, test_error, values, pred_df, error_list = model.test(data_loader)

    "save predictions"
    pred_df.to_csv(cfg.output_directory + "hiclstm_%s_predictions_chr%s.csv" % (cfg.cell, str(chr)), sep="\t")


if __name__ == '__main__':
    cfg = config.Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "Initalize Model"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    for chr in cfg.chr_test_list:
        test_model(model, cfg, chr)
