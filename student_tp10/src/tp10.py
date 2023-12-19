# %%
import logging
import math
import re
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamaestro import prepare_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return (
            self.tokenizer(s if isinstance(s, str) else s.read_text()),
            self.filelabels[ix],
        )

    def get_txt(self, ix):
        s = self.files[ix]
        return s if isinstance(s, str) else s.read_text(), self.filelabels[ix]


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        "edu.stanford.glove.6b.%d" % embedding_size
    ).load()
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return (
        word2id,
        embeddings,
        FolderText(ds.train.classes, ds.train.path, tokenizer, load=False),
        FolderText(ds.test.classes, ds.test.path, tokenizer, load=False),
    )


# %%


#  TODO:
class SelfAttentionModel(nn.Module):
    def __init__(self, input_size, hidd_size, output_size, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidd_size = hidd_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Couches Q, K, V
        self.linear_q = nn.Linear(self.input_size, self.hidd_size)
        self.linear_k = nn.Linear(self.input_size, self.hidd_size)
        self.linear_v = nn.Linear(self.input_size, self.hidd_size)

        # Couche de sortie
        self.linear_out = nn.Linear(self.hidd_size, self.output_size)


# %%
@click.command()
@click.option(
    "--test-iterations",
    default=1000,
    type=int,
    help="Number of training iterations (batches) before testing",
)
@click.option("--epochs", default=50, help="Number of epochs.")
@click.option(
    "--modeltype",
    required=True,
    type=int,
    help="0: base, 1 : Attention1, 2: Attention2",
)
@click.option("--emb-size", default=100, help="embeddings size")
@click.option("--batch-size", default=20, help="batch size")
def main(epochs, test_iterations, modeltype, emb_size, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    def collate(batch):
        """Collate function for DataLoader"""
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return (
            emb_layer(
                torch.nn.utils.rnn.pad_sequence(
                    data, batch_first=True, padding_value=PAD
                )
            ).to(device),
            torch.LongTensor(labels).to(device),
            torch.Tensor(lens).to(device),
        )

    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate, shuffle=False
    )
    ##  TODO:
    model = SelfAttentionModel(emb_size, 100, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter("runs/tp10")
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels, lens) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs, lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels, lens in test_loader:
                outputs = model(inputs, lens)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            writer.add_scalar("Accuracy/test", 100 * correct / total, epoch)
            print("Accuracy: %d %%" % (100 * correct / total))
    writer.flush()
    writer.close()


# %%

if __name__ == "__main__":
    main()

# %%
