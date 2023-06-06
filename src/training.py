import datetime
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
from src.data_processing import SDDataset, process_gcnr_data, process_gcnsi_data
import torch
from tqdm import tqdm
import src.constants as const
from src import utils


def train(model, dataset, criterion):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
    :return: The trained model.
    """

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)
    epochs = range(1, const.EPOCHS)
    losses = []
    print("Train Model:")
    for epoch in tqdm(epochs):
        running_loss = 0.0
        for data in dataset:
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
    return model


def main():
    """Initiates the training of the classifier specified in the constants file."""

    print("Prepare data ...")

    if const.MODEL == "GCNSI":
        model = GCNSI()
        train_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnsi_data)[
            : const.TRAINING_SIZE
        ]
        criterion = torch.nn.BCEWithLogitsLoss()

    elif const.MODEL == "GCNR":
        model = GCNR()
        train_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnr_data)[
            : const.TRAINING_SIZE
        ]
        criterion = torch.nn.MSELoss()

    train(model, train_data, criterion)

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    utils.save_model(model, f"{const.MODEL}_{current_time}")
    utils.save_model(model, f"{const.MODEL}_latest")


if __name__ == "__main__":
    main()
