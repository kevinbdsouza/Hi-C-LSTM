import torch
import time
from torch.utils.tensorboard import SummaryWriter
from training.alt.alt_config import Config
from training.alt.alt_model import SeqLSTM
from training.alt.alt_data_utils import get_data_loader_batch_chr, save_processed_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(cfg, writer):
    """
    train_model(cfg, model_name, writer) -> No return object
    Loads existing model (or creates new one), loads data, trains the model.
    Saves the trained model in a .pth file.
    Specify training and model parameters in the configuration file.
    Args:
        cfg (Config): the configuration to use for the experiment.
        model_name (string): The model name that needs to be used to load model or create new model
        writer (SummaryWriter): tensorboard summary writer
    """

    "Initalize model and load model weights if they exist"
    model = SeqLSTM(cfg, device).to(device)
    model.load_weights()

    "Initalize optimizer"
    optimizer = model.compile_optimizer()

    "Train model"
    model.train_model(optimizer, writer)

    "Save model"
    torch.save(model.state_dict(), cfg.model_dir + cfg.model_name + '.pth')


if __name__ == '__main__':
    cfg = Config()
    cell = cfg.cell
    model_name = cfg.model_name

    "Set up Tensorboard logging"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('./tensorboard_logs/' + model_name + timestr)

    if cfg.save_processed_data:
        "Process input data and save the input IDs and Hi-C Values"
        save_processed_data(cfg)

    "Train the model and save the .pth file"
    train_model(cfg, writer)
