import torch
from functools import partial
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torchtext.vocab import Vocab
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utils.config import ModelConfig
from utils.model import CBOWModel, SkipGramModel


class Word2VecDataLoader:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build_vocab(self, data_iter: Dataset, tokenizer):
        """Builds vocabulary from iterator

        Args:
            tokenizer (): callable tokenizer, pass in a sentence and return an
            array of vocabulary.
            data_iter: the dataset.

        Returns: vocabulary build from the dataset.

        """

        vocab = build_vocab_from_iterator(
            map(tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=self.config.min_word_freq,
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def collate_cbow(self, batch: list, text_pipeline):
        """
        Collate_fn for CBOW model to be used with Dataloader.
        `batch` is expected to be list of text paragrahs.

        Context is represented as N=CBOW_N_WORDS past words
        and N=CBOW_N_WORDS future words.

        Long paragraphs will be truncated to contain
        no more that MAX_SEQUENCE_LENGTH tokens.

        Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
        Each element in `batch_output` is a middle word.

        Args:
            text_pipeline (): callable preprocessor to convert a document to
            an array of indexed vocabularies.
            batch: a batch of documents.

        Returns: batched input and output tensor.

        """
        batch_input, batch_output = [], []
        for text in batch:
            # Preprocess the document to get indexed document.
            text_tokens_ids = text_pipeline(text)

            # If the document is too short to generate a training data.
            if len(text_tokens_ids) < self.config.n_words * 2 + 1:
                continue

            # Crop the document.
            if self.config.max_seq_length:
                text_tokens_ids = text_tokens_ids[:self.config.max_seq_length]

            # Generate a batch
            for idx in range(len(text_tokens_ids) - self.config.n_words * 2):
                # Get the sequence.
                token_id_sequence = text_tokens_ids[idx: (
                    idx + self.config.n_words * 2 + 1)]
                # Get the center word as the output.
                output = token_id_sequence.pop(self.config.n_words)
                # Get the rest as input sequence.
                input_ = token_id_sequence
                # Insert the data into the batch.
                batch_input.append(input_)
                batch_output.append(output)

        # Convert to tensor.
        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output

    def collate_skipgram(self, batch: list, text_pipeline):
        """
        Collate_fn for Skip-Gram model to be used with Dataloader.
        `batch` is expected to be list of text paragrahs.

        Context is represented as N=SKIPGRAM_N_WORDS past words
        and N=SKIPGRAM_N_WORDS future words.

        Long paragraphs will be truncated to contain
        no more that MAX_SEQUENCE_LENGTH tokens.

        Each element in `batch_input` is a middle word.
        Each element in `batch_output` is a context word.

        Args:
            text_pipeline (): callable preprocessor to convert a document to
            an array of indexed vocabularies.
            batch: a batch of documents.

        Returns: batched input and output tensor.

        """
        batch_input, batch_output = [], []
        for text in batch:
            # Preprocess the document to get indexed document.
            text_tokens_ids = text_pipeline(text)

            # If the document is too short to generate a training data.
            if len(text_tokens_ids) < self.config.n_words * 2 + 1:
                continue

            # Crop the document.
            if self.config.max_seq_length:
                text_tokens_ids = text_tokens_ids[:self.config.max_seq_length]

            # Generate a batch
            for idx in range(len(text_tokens_ids) - self.config.n_words * 2):
                # Get the sequence.
                token_id_sequence = text_tokens_ids[idx: (
                    idx + self.config.n_words * 2 + 1)]
                # Get the center word as the output.
                input_ = token_id_sequence.pop(self.config.n_words)
                # Get the rest as input sequence.
                output_ = token_id_sequence
                # Insert the data into the batch.
                for output in output_:
                    batch_input.append(input_)
                    batch_output.append(output)

        # Convert to tensor.
        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output

    def get_dataloader_and_vocab(self, model, data_iter: Dataset, batch_size: int, shuffle: bool, vocab: Vocab = None):
        """Get the dataloader with the model, dataset, and optional vocabulary.

        Args:
            model (): the model used for the training.
            data_iter: the dataset.
            batch_size: the batch size.
            shuffle: If the the DataLoader requires shuffle.
            vocab: the optional vocabulary used for text_pipeline preprocessing
            If None, build the vocabulary from the dataset.

        Returns: a DataLoader and a vocabulary

        """
        tokenizer = get_tokenizer("basic_english", language="en")

        if not vocab:
            vocab = self.build_vocab(data_iter, tokenizer)

        def text_pipeline(x): return vocab(tokenizer(x))

        if isinstance(model, CBOWModel):
            collate_fn = self.collate_cbow
        elif isinstance(model, SkipGramModel):
            collate_fn = self.collate_skipgram
        else:
            raise ValueError("Unsupported model.")

        dataloader = DataLoader(
            data_iter,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
        )
        return dataloader, vocab
