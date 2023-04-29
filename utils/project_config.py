from dataclasses import dataclass

import torch.nn as nn

from utils.model import ModelType


@ dataclass
class ProjectConfig:
    use_cuda: bool = False

    model_type: ModelType = ModelType.SKIPGRAM
    criterion = nn.CrossEntropyLoss

    learning_rate: float = 0.025
    epochs: int = 16
    train_steps: int = -1
    val_steps: int = -1

    data_dir: str = "data/"
    model_dir: str = "models/"
    train_batch_size: int = 16
    val_batch_size: int = 16
    shuffle: bool = True
    checkpoint_freq: int = 4
