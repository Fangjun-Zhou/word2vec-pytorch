class ModelConfig:
    def __init__(
        self,
        n_words=4,
        min_word_freq=50,
        max_seq_length=256,
        embed_dim=256,
        embed_max_norm=1,
    ):
        self.n_words = n_words
        self.min_word_freq = min_word_freq
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.embed_max_norm = embed_max_norm
