""" Initiates the training of the classifier specified in the constants file. """
import datetime
from typing import Tuple
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import torch
from tqdm import tqdm
import src.constants as const
from src import utils
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from src.data_processing import SDDataset
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        pred[pred < 0] = 0
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


def subsampleClasses(
    y: torch.Tensor, y_hat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # subsample the majority class
    source_indicator = 0 if const.MODEL == "GCNR" else 1
    non_sources = torch.where(y != source_indicator)[0]
    sources = torch.where(y == source_indicator)[0]
    random_numbers = torch.randperm(non_sources.shape[0])[: sources.shape[0]]
    subsampled_non_sources = non_sources[random_numbers]
    indices = torch.cat((subsampled_non_sources, sources))
    return y[indices], y_hat[indices]


def train(
    model: torch.nn.Module,
    model_name: str,
    dataset: SDDataset,
    criterion: torch.nn.Module,
):
    """
    Trains the specified model using the given dataset.
    :param model: The model to train.
    :param model_name: The name of the model.
    :param dataset: The dataset to train on, containing the graph structure, features, and labels.
    :param criterion: The loss criterion used for training.
    :return: The trained model.
    """
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)
    epochs = range(1, const.EPOCHS)
    print(f"Train Model on device:{device} :")
    min_loss = float("inf")
    loader = DataListLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=True)
    for epoch in tqdm(epochs, disable=const.ON_CLUSTER):
        agg_loss = 0
        for data_list in loader:
            optimizer.zero_grad()
            out = model(data_list)
            y = torch.cat([data.y for data in data_list]).to(out.device)
            if const.SUBSAMPLE:
                y, out = subsampleClasses(y, out)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            agg_loss += loss.item()

        print(f"Epoch: {epoch}\tLoss: {agg_loss:.4f}")
        writer.add_scalar("Loss/train", agg_loss, epoch)
        if agg_loss < min_loss:
            print("Saving new best model ...")
            min_loss = agg_loss
            utils.save_model(model.module, "latest")
            utils.save_model(model.module, model_name)
    writer.flush()
    writer.close()
    return model


def main():
    """
    Initiates the training of the classifier specified in the constants file.
    """
    print("Prepare Data ...")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = (
        f"{const.MODEL}_{current_time}"
        if const.MODEL_NAME is None
        else const.MODEL_NAME
    )

    if const.MODEL == "GCNR":
        model = GCNR()
        criterion = MSLELoss() if const.USE_LOG_LOSS else torch.nn.MSELoss()

    elif const.MODEL == "GCNSI":
        model = GCNSI()
        criterion = torch.nn.BCEWithLogitsLoss()

    train_data = utils.load_processed_data("synthetic", False)
    train(model, model_name, train_data, criterion)


if __name__ == "__main__":
    main()
