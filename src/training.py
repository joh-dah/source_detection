""" This file contains the main function for training the model. """

import datetime
import architectures.GCNSI_model as gcnsi
import torch
from tqdm import tqdm
import src.constants as const
import src.GCN_model as gcn
import src.GCNR_model as gcnr
from src import utils


def train(model, data):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
    :return: The trained model.
    """
    epochs = range(1, const.EPOCHS)
    losses = []
    print("Train Model:")
    for epoch in tqdm(epochs):
        running_loss = 0.0
        for graph_structure, features, labels in data:
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=const.WEIGHT_DECAY
            )
            optimizer.zero_grad()
            out = model(features, graph.edge_index)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            losses.append(loss)
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
    return model


def main():
    """Initiates the training of the classifier specified in the constants file."""

    train_data = utils.load_data(const.DATA_PATH + "/train")

    if const.MODEL == "GCNSI":
        model = gcn.GCN()
        criterion = torch.nn.BCEWithLogitsLoss()

    elif const.MODEL == "GCNR":
        model = gcnr.GCNR()
        criterion = torch.nn.MSELoss()

    train(model, train_data)

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    utils.save_model(model, f"{const.MODEL}_{current_time}")
    utils.save_model(model, f"{const.MODEL}_latest")


if __name__ == "__main__":
    main()
