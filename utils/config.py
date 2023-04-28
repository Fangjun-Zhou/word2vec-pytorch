class ModelConfig:
    def __init__(
        self,
        nWords=4,
        minWordFreq=50,
        maxSeqLength=256,
        embedDim=256,
        embedMaxNorm=1,
    ):
        self.nWords = nWords
        self.minWordFreq = minWordFreq
        self.maxSeqLength = maxSeqLength
        self.embedDim = embedDim
        self.embedMaxNorm = embedMaxNorm
