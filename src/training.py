import datetime
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
from src.data_processing import (
    SDDataset,
    process_gcnr_data,
    process_gcnsi_data,
    process_simplified_gcnsi_data,
)
import torch
from tqdm import tqdm
import src.constants as const
from src import utils
from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


def train(model, model_name, dataset, criterion):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
    :return: The trained model.
    """

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)
    epochs = range(1, const.EPOCHS)
    print(f"Train Model on device:{device} :")
    min_loss = float("inf")
    loader = DataLoader(dataset, batch_size=const.BATCH_SIZE)
    for epoch in tqdm(epochs):
        agg_loss = 0
        for data in loader:
            data.to(device)
            x = data.x
            y = data.y
            edge_index = data.edge_index
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            agg_loss += loss.item()
        if agg_loss < min_loss:
            print(f"Epoch: {epoch}\tLoss: {agg_loss:.4f}")
            print("Saving new best model ...")
            min_loss = agg_loss
            utils.save_model(model, "latest")
            utils.save_model(model, model_name)
    return model


def main():
    """Initiates the training of the classifier specified in the constants file."""

    print("Prepare Data ...")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = f"{const.MODEL}_{current_time}_{const.MODEL_NAME}"

    if const.MODEL == "GCNR":
        model = GCNR().to(device)
        criterion = MSLELoss() if const.USE_LOG_LOSS else torch.nn.MSELoss()

    elif const.MODEL == "GCNSI":
        model = GCNSI().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

    train_data = utils.load_processed_data("train")
    train(model, model_name, train_data, criterion)


if __name__ == "__main__":
    main()
