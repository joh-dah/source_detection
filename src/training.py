import src.constants as const
import src.GCN_model as gcn
import src.utils as utils
import datetime


def main():
    """Initiates the training of the classifier specified in the constants file."""

    train_data = utils.load_data(const.DATA_PATH + "/train")

    if const.MODEL == "GCN":
        model = gcn.GCN()
        train_data = gcn.prepare_data(train_data)
        model = gcn.train(model, train_data)

    elif const.MODEL == "GCNSI":
        # TODO: implement
        pass

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    utils.save_model(model, f"{const.MODEL}_{current_time}")
    utils.save_model(model, f"{const.MODEL}_latest")


if __name__ == "__main__":
    main()
