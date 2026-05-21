from pixel_sprite.controller.train import train_model
from pixel_sprite.model.loss import PixelArtLoss

__all__ = ["PixelArtLoss", "train_model"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train pixel sprite direction transfer GAN model.")
    parser.add_argument(
        "-v", "--version",
        type=int,
        default=2,
        choices=[1, 2],
        help="Model architecture version (1 or 2, default is 2)"
    )
    args = parser.parse_args()
    train_model(version=args.version)
