from dataclasses import dataclass


@ dataclass
class ModelConfig:
    n_words: int = 4
    min_word_freq: int = 64
    max_seq_length: int = 256
    embed_dim: int = 256
    embed_max_norm: float = 1
