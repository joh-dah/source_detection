import src.constants as const
import src.GCN_model as gcn
import src.utils as utils


def main():
    """Initiates the validation of the classifier specified in the constants file."""

    val_data = utils.load_data(const.DATA_PATH + "/validation")

    if const.MODEL == "GCN":
        model = gcn.GCN()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        prep_val_data = gcn.prepare_data(val_data)
        gcn.evaluate(model, prep_val_data)
        gcn.vizualize_results(model, val_data[:5])


if __name__ == "__main__":
    main()
