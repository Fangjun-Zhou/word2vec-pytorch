import os

import torch

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torchtext.vocab import Vocab


def getLrScheduler(optimizer: optim.Optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.

    Args:
        optimizer: the optimizer used for the training.
        total_epochs: the total number of epochs.
        verbose: verbose enable.

    Returns: LambdaLR scheduler.

    """
    def lrLambda(epoch): return (total_epochs - epoch) / total_epochs
    lrScheduler = LambdaLR(optimizer, lr_lambda=lrLambda, verbose=verbose)
    return lrScheduler


def saveVocab(vocab: Vocab, modelDir: str):
    """Save vocab file to `model_dir` directory.

    Args:
        vocab: the vocab to save.
        model_dir: the directory to store the vocab.
    """
    vocabPath = os.path.join(modelDir, "vocab.pt")
    torch.save(vocab, vocabPath)


def loadVocab(modelDir: str):
    """Load vocab file from `model_dir` directory.

    Args:
        model_dir: the directory to store the vocab.

    Returns: Loaded vocab.

    """
    """Load vocab file from `model_dir` directory"""
    vocab = torch.load(os.path.join(modelDir, "vocab.pt"))
    return vocab
