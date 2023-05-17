import src.constants as const
import src.GCN_model as gcn
import src.GCNSI_model as gcnsi
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
    elif const.CLASSIFIER == "GCNSI":
        model = gcnsi.GCNSI()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.CLASSIFIER}_latest.pth")
        val_data = gcnsi.prepare_data(val_data)
        gcn.evaluate(model, val_data)


if __name__ == "__main__":
    main()
