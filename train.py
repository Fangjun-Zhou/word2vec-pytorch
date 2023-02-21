import argparse
import yaml
import os
import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split

from utils.dataloader import get_dataloader_and_vocab, get_custom_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)


def train(
    config,
    data_iter=None,
    vocab: Vocab=None,
    transfer_model: nn.Module=None
):
    # Check current device.
    if (config["use_cuda"] and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("Using device:", device)
    
    if data_iter is None:
        train_dataloader, vocab = get_dataloader_and_vocab(
            model_name=config["model_name"],
            ds_name=config["dataset"],
            ds_type="train",
            data_dir=config["data_dir"],
            batch_size=config["train_batch_size"],
            shuffle=config["shuffle"],
            vocab=None,
        )

        val_dataloader, _ = get_dataloader_and_vocab(
            model_name=config["model_name"],
            ds_name=config["dataset"],
            ds_type="valid",
            data_dir=config["data_dir"],
            batch_size=config["val_batch_size"],
            shuffle=config["shuffle"],
            vocab=vocab,
        )
    else:
        # Split the dataset into train and validation sets.
        train_dataset, val_dataset = train_test_split(data_iter, test_size=0.2)
        train_dataloader, vocab = get_custom_dataloader_and_vocab(
            model_name=config["model_name"],
            data_iter=train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=config["shuffle"],
            vocab=vocab
        )
        val_dataloader, _ = get_custom_dataloader_and_vocab(
            model_name=config["model_name"],
            data_iter=val_dataset,
            batch_size=config["val_batch_size"],
            shuffle=config["shuffle"],
            vocab=vocab
        )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    
    # If transfer learning is enabled, load the pre-trained model.
    if transfer_model:
        weight = model.embeddings.weight.detach().numpy()
        transfer_weight = transfer_model["embeddings.weight"].numpy()
        # Check if the vocabulary of transfer model is larger than the current model.
        if transfer_weight.shape[0] > weight.shape[0]:
            # If so, use the first part of the transfer model's vocabulary.
            weight = transfer_weight[:weight.shape[0], :]
        else:
            # Otherwise, use the first part of the current model's vocabulary.
            weight[:transfer_weight.shape[0], :] = transfer_weight
        # Load the weight to the current model.
        with torch.no_grad():
            model.embeddings.weight.copy_(torch.from_numpy(weight))
        print("Transfer learning enabled. Pre-trained model loaded.")
        

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)