from argparse import ArgumentParser
from trainer import Transformer_pl
from utils import Config

def main(args):
    config = Config.load("./config.json")
    model = Transformer_pl(config).load_from_checkpoint(args.model_saved_path)

    print("Enter QUIT for quit")
    input_kor = input("kor : ")
    while input_kor != "QUIT":
        print("eng : ", model.translate(input_kor))
        input_kor = input("kor : ")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Set path
    parser.add_argument('--model_saved_path', type=str, default="./model_saved/set/your/checkpoint")
    args = parser.parse_args()

    main(args)