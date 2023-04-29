class ModelConfig:
    def __init__(
        self,
        nWords=4,
        minWordFreq=50,
        maxSeqLength=256,
        embedDim=256,
        embedMaxNorm=1,
    ):
        self.n_words = nWords
        self.min_word_freq = minWordFreq
        self.max_seq_length = maxSeqLength
        self.embed_dim = embedDim
        self.embed_max_norm = embedMaxNorm
