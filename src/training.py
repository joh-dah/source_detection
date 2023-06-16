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


def train(model, model_name, dataset, criterion):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
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
    """Initiates the training of the classifier specified in the constants file."""

    print("Prepare Data ...")
    print(const.GCNSI_N_FEATURES)

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    model_name = f"{const.MODEL}_{current_time}"

    if const.MODEL == "GCNSI":
        model = GCNSI().to(device)
        train_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnsi_data)[
            : const.TRAINING_SIZE
        ]
        criterion = torch.nn.BCEWithLogitsLoss()

    elif const.MODEL == "SMALL_INPUT_GCNSI":
        model = GCNSI().to(device)
        train_data = SDDataset(
            const.DATA_PATH, pre_transform=process_simplified_gcnsi_data
        )[: const.TRAINING_SIZE]
        criterion = torch.nn.BCEWithLogitsLoss()

    elif const.MODEL == "GCNR":
        model = GCNR().to(device)
        train_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnr_data)[
            : const.TRAINING_SIZE
        ]
        criterion = torch.nn.MSELoss()

    train(model, model_name, train_data, criterion)
    # utils.save_model(model, "latest")
    # utils.save_model(model, model_name)


if __name__ == "__main__":
    main()
