import os

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from utils.dataloader import get_custom_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_vocab,
    load_vocab
)

# ------------------------------------------------------------ #
#                           Settings                           #
# ------------------------------------------------------------ #

MODEL_DIR = os.path.join(*["models", "skipgram"])
MODEL_NAME = "skipgram"

BATCH_SIZE = 96
SHUFFLE = True
DATA_SET_SIZE = 20000

OPTIMIZER = "Adam"
LEARNING_RATE = 0.025
EPOCHS = 64
TRAIN_STEPS = None
VAL_STEPS = None

CHECKPOINT_FREQUENCY = 4

# PRE_TRAINED_MODEL_PATH = os.path.join(*["weights", "skipgram_WikiText2", "model.pt"])
# PRE_TRAINED_VOCAB_PATH = os.path.join(*["weights", "skipgram_WikiText2", "vocab.pt"])
PRE_TRAINED_MODEL_PATH = None
PRE_TRAINED_VOCAB_PATH = None

USE_CUDA = True

# Check current device.
if (USE_CUDA and torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Using device: {}".format(device))

# ------------------------------------------------------------ #
#                           Load Data                          #
# ------------------------------------------------------------ #

# Dataset for blog text.
class BlogDataset(Dataset):
    def __init__(self, df, size = -1):
        self.blog_df = df
        # Shuffle and take a subset of the data.
        if size > 0:
            self.blog_df = self.blog_df.sample(frac=1).reset_index(drop=True)
            self.blog_df = self.blog_df[:size]
        
    def __len__(self):
        return len(self.blog_df)
    
    def __getitem__(self, idx):
        return self.blog_df.iloc[idx, 6]

blogDf = pd.read_csv(os.path.join(*["data", "blog.zip"]))
print("Dataset size: {}".format(len(blogDf)))

# Read in the datset.
blog_dataset = BlogDataset(blogDf, size=DATA_SET_SIZE)
# Split the dataset into train and validation sets.
train_dataset, val_dataset = train_test_split(blog_dataset, test_size=0.2)

print("Train dataset size: {}".format(len(train_dataset)))
print("Validation dataset size: {}".format(len(val_dataset)))

if (PRE_TRAINED_VOCAB_PATH):
    vocab = load_vocab(PRE_TRAINED_VOCAB_PATH)
else:
    vocab = None

train_loader, vocab = get_custom_dataloader_and_vocab(
    model_name=MODEL_NAME,
    data_iter=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    vocab=vocab
)
test_loader, _ = get_custom_dataloader_and_vocab(
    model_name=MODEL_NAME,
    data_iter=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    vocab=vocab
)

vocab_size = len(vocab.get_stoi())

print("Vocabulary size: {}".format(vocab_size))

# ------------------------------------------------------------ #
#                             Train                            #
# ------------------------------------------------------------ #

model_class = get_model_class(MODEL_NAME)
model = model_class(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer_class = get_optimizer_class(OPTIMIZER)
optimizer = optimizer_class(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_lr_scheduler(optimizer, EPOCHS, verbose=True)

trainer = Trainer(
    model=model,
    epochs=EPOCHS,
    train_dataloader=train_loader,
    train_steps=TRAIN_STEPS,
    val_dataloader=test_loader,
    val_steps=VAL_STEPS,
    criterion=criterion,
    optimizer=optimizer,
    checkpoint_frequency=CHECKPOINT_FREQUENCY,
    lr_scheduler=lr_scheduler,
    device=device,
    model_dir=MODEL_DIR,
    model_name=MODEL_NAME,
)

if PRE_TRAINED_MODEL_PATH:
    trainer.load_model(PRE_TRAINED_MODEL_PATH)

trainer.train()
print("Training finished.")

trainer.save_model()
trainer.save_loss()
save_vocab(vocab, MODEL_DIR)
print("Model artifacts saved to folder:", MODEL_DIR)