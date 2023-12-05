# %%
import logging
from typing import Any

from torch.nn.modules.pooling import MaxPool1d

logging.basicConfig(level=logging.INFO)

import gzip
import heapq
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tp8_preprocess import TextDataset
from tqdm import tqdm

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)


def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE = 500
TEST_BATCHSIZE = 500

# %%
# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(
    train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate
)
val_iter = torch.utils.data.DataLoader(
    val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate
)
test_iter = torch.utils.data.DataLoader(
    test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate
)


#  TODO:
data = next(iter(train_iter))
print(data)
# %%
len(data[1])
# %%
# Convert data[0] to Float
X = data[0].float()
data[0].shape


# %%
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        # scope : parcours les couches, if conv1D -> self.scope(s,k), puis pooling m:stride
        self.emb = (nn.Embedding(vocab_size, embedding_dim, padding_idx=0),)
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            MaxPool1d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 2, 1)  # x : [batch, emb, seq_len]
        return self.model(x)


# %%
class Scope:
    """
    Calcule les tailles W, S pour les données
    """

    def __init__(self):
        pass

    def __call__(self) -> Any:
        """
        Formules de récurrence
        """
        pass


# %%
class Sample:
    """
    1 exemple forward
    indice debut - indice fin
    """


# %%
X.shape
# %%
conv = nn.Conv1d(20, 100, kernel_size=3, stride=1, padding=1)

# %%
emb = nn.Embedding(1000, 20)
# %%
d = emb(data[0])
d = d.permute(0, 2, 1)
d.shape


# %%
X1 = X.unsqueeze(1)
X1.shape
# %%
conv(d)
