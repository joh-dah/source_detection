""" This file contains the code for validating the model. """

import src.constants as const
import src.GCN_model as gcn
import src.GCNSI_model as gcnsi
from src import utils


def main():
    """Initiates the validation of the classifier specified in the constants file."""

    val_data = utils.load_data(const.DATA_PATH + "/validation")

    n_plots = 5
    prep_val_data = None

    if const.MODEL == "GCN":
        model = gcn.GCN()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        prep_val_data = gcn.prepare_data(val_data)

    elif const.MODEL == "GCNSI":
        model = gcnsi.GCNSI()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        prep_val_data = gcnsi.prepare_data(val_data)

    utils.evaluate(model, prep_val_data)
    utils.vizualize_results(model, val_data[:n_plots], prep_val_data[:n_plots])


if __name__ == "__main__":
    main()
