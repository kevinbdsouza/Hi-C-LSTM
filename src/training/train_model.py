import torch
import time
from torch.utils.tensorboard import SummaryWriter
import training.config as config
from training.model import SeqLSTM
from training.data_utils import get_data_loader_batch_chr, save_processed_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(cfg, model_name, writer):
    '''
    :param cfg:
    :param model_name:
    :param writer:
    :return:
    '''

    "Initalize model and load model wrights if they exist"
    model = SeqLSTM(cfg, device, model_name).to(device)
    model.load_weights()

    "Initalize optimizer"
    optimizer, criterion = model.compile_optimizer(cfg)

    "Get data"
    data_loader, samples = get_data_loader_batch_chr(cfg)
    print("%s batches loaded" % str(len(data_loader)))

    "Train model"
    model.train_model(data_loader, criterion, optimizer, writer)

    "Save model"
    torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')


if __name__ == '__main__':
    cfg = config.Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "Set up Tensorboard logging"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('./tensorboard_logs/' + model_name + timestr)

    "Process input data and save the input IDs and Hi-C Values"
    save_processed_data(cfg, cell)

    "Train the model and save the .pth file"
    train_model(cfg, model_name, writer)
