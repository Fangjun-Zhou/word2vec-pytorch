from enum import Enum

import torch.nn as nn

from utils.config import ModelConfig


class ModelType(Enum):
    CBOW = 1
    SKIPGRAM = 2


class CBOWModel(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, config: ModelConfig):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embed_dim,
            max_norm=config.embed_max_norm,
        )
        self.linear = nn.Linear(
            in_features=config.embed_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGramModel(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, config: ModelConfig):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.embed_dim,
            max_norm=config.embed_max_norm,
        )
        self.linear = nn.Linear(
            in_features=config.embed_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
