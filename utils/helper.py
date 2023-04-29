import os

import torch

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torchtext.vocab import Vocab


def get_lr_scheduler(optimizer: optim.Optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.

    Args:
        optimizer: the optimizer used for the training.
        total_epochs: the total number of epochs.
        verbose: verbose enable.

    Returns: LambdaLR scheduler.

    """
    def lr_lambda(epoch): return (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_vocab(vocab: Vocab, model_dir: str):
    """Save vocab file to `model_dir` directory.

    Args:
        vocab: the vocab to save.
        model_dir: the directory to store the vocab.
    """
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)


def load_vocab(model_dir: str):
    """Load vocab file from `model_dir` directory.

    Args:
        model_dir: the directory to store the vocab.

    Returns: Loaded vocab.

    """
    """Load vocab file from `model_dir` directory"""
    vocab = torch.load(os.path.join(model_dir, "vocab.pt"))
    return vocab
