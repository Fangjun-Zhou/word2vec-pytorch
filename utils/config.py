class ModelConfig:
    def __init__(
        self,
        n_words=4,
        min_word_freq=50,
        max_seq_length=256,
        embed_dim=256,
        embed_max_norm=1,
    ):
        # Number of words include in the context.
        self.n_words = n_words
        # The minimum word frequency to build the vocab.
        self.min_word_freq = min_word_freq
        # The max length of a sequence to preprocess the dataset.
        self.max_seq_length = max_seq_length
        # The embedded word vector dimension.
        self.embed_dim = embed_dim
        # The max norm of the embedded vector.
        self.embed_max_norm = embed_max_norm
