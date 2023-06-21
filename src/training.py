""" Initiates the training of the classifier specified in the constants file. """
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model: torch.nn.Module, model_name: str, dataset: SDDataset, criterion: torch.nn.Module):
    """
    Trains the specified model using the given dataset.
    :param model: The model to train.
    :param model_name: The name of the model.
    :param dataset: The dataset to train on, containing the graph structure, features, and labels.
    :param criterion: The loss criterion used for training.
    :return: The trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)
    epochs = range(1, const.EPOCHS)
    losses = []
    print(f"Train Model on device:{device} :")
    min_loss = float("inf")
    for epoch in tqdm(epochs):
        running_loss = 0.0
        for data in dataset:
            data.to(device)
            x = data.x
            y = data.y
            edge_index = data.edge_index
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss
            losses.append(loss)
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
        if running_loss < min_loss:
            print("Saving new best model ...")
            min_loss = running_loss
            utils.save_model(model, "latest")
            utils.save_model(model, model_name)
    return model


def main():
    """
    Initiates the training of the classifier specified in the constants file.
    """
    print("Prepare Data ...")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = f"{const.MODEL}_{current_time}"

    if const.MODEL == "GCNR":
        model = GCNR().to(device)
        criterion = torch.nn.MSELoss()

    elif const.MODEL == "GCNSI":
        model = GCNSI().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

    train_data = utils.load_processed_data("train")
    train(model, model_name, train_data, criterion)


if __name__ == "__main__":
    main()
