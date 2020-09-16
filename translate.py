from argparse import ArgumentParser
from trainer import Transformer_pl
from utils import Config


def main(args):
    config = Config.load("./config.json")
    model = Transformer_pl(config).load_from_checkpoint(args.model_saved_path)

    print("종료를 원하시면 EXIT를 입력하세요")
    input_kor = input("kor : ")
    while input_kor != "EXIT":
        print("eng : ", model.translate(input_kor))
        input_kor = input("kor : ")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Training
    parser.add_argument('--model_saved_path', type=str, default="./model_saved/epoch=25-val_loss=1.3987.ckpt")
    args = parser.parse_args()
    main(args)


