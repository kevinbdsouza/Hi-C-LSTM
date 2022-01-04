import torch
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_chr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_model(model, cfg, cell, chr):
    # get data
    data_loader, samples = get_data_loader_chr(cfg, chr)

    # test model
    predictions, test_error, values, pred_df, error_list = model.test(data_loader)

    pred_df.to_csv(cfg.output_directory + "shuffle_%s_predictions_chr%s.csv" % (cell, str(chr)), sep="\t")
    # np.save(cfg.output_directory + "shuffle_renew_zero_chr%s.npy" % str(chr), zero_embed)


if __name__ == '__main__':
    # test_chr = list(range(1, 11))
    test_chr = list(range(15, 23))
    # test_chr = [21]
    cfg = config.Config()
    cell = cfg.cell
    model_name = "shuffle_" + cell

    # initalize model
    model = SeqLSTM(cfg, device, model_name).to(device)

    # load model weights
    model.load_weights()

    for chr in test_chr:
        test_model(model, cfg, cell, chr)

print("done")
