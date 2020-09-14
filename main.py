from argparse import ArgumentParser
from utils import Config
from trainer import Transformer_pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(42)

def main(args):
    config = Config.load("./config.json")
    checkpoint_callback = ModelCheckpoint(
        filepath="./model_saved/{epoch}-{val_loss:.4f}",
        save_top_k=True,
        verbose=True,
        monitor="val_loss",
        mode="min")

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Transformer_pl(hparams=config)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(gpus=args.gpus,
                      max_epochs=args.epochs,
                      checkpoint_callback=checkpoint_callback,
                      gradient_clip_val=1)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Training
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    main(args)
