from pixel_sprite.controller.train import train_model
from pixel_sprite.model.loss import PixelArtLoss

__all__ = ["PixelArtLoss", "train_model"]


if __name__ == "__main__":
    train_model()
