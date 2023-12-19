# %%
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamaestro import prepare_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

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
data = get_imdb_data()
from torch.nn.utils.rnn import pad_sequence

# %%
data[2].__getitem__(1)


# %%
def collate_fn(batch):
    """Collate using pad_sequence"""
    data = [torch.tensor(item[0]) for item in batch]
    data = pad_sequence(data, batch_first=True)
    labels = torch.tensor([item[1] for item in batch])
    lengths = torch.tensor([len(item[0]) for item in batch])
    return data, labels, lengths


# Dataloader
train_loader = DataLoader(data[2], batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(data[3], batch_size=32, shuffle=True, collate_fn=collate_fn)
# %%
x, y, len = next(iter(train_loader))
# %%
x, y, len


# %%
def masked_softmax(x, mask):
    x = x - x.max(dim=1, keepdim=True)[0]
    x = x.exp()
    x = x * mask
    x = x / x.sum(dim=1, keepdim=True)
    return x


# Example of masked softmax
x = torch.tensor([[1, 2, 0], [4, 5, 6]], dtype=torch.float32)
mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
print(masked_softmax(x, mask))


# %%
# Classification de texte (pos/neg).
# Modèle de base. Un texte t = (t1, ..., tn) est représenté par la moyenne des embeddings des mots.
# t_hat = 1/n sum_i=1^n x_i
# On calcule ensuite une couche linéaire pour prédire la classe.
class TextClassifier(nn.Module):
    def __init__(self, emb):
        super(TextClassifier, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(emb)
        self.linear = nn.Linear(emb.shape[1], 2)

    def forward(self, x):
        """x is a tensor of shape (seq_len, batch_size)"""
        x = self.embeddings(x)
        x = self.attention(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

    def attention(self, x):
        """Attention sur les mots
        Ici modèle de base: même poids pour tous les mots
        """
        # Retourne le softmax masqué de x
        return masked_softmax(x, (x != 0).float())


# %%
model = TextClassifier(torch.tensor(data[1], dtype=torch.float32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()

# %%
for epoch in range(10):
    model.train()
    for x, y, len in tqdm(train_loader):
        optimizer.zero_grad()
        y_hat = model(x)
        print(y_hat.shape, y.shape)
        # y_hat shape is (batch_size, 2)
        # y shape is (1, batch_size)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(test_loader):
        y_hat = model(x.T)
        correct += (y_hat.argmax(dim=1) == y).sum().item()
        total += y.shape[0]

    print("Epoch %d: %f" % (epoch, correct / total))


# %%
for batch in train_loader:
    x, y, length = batch
# %%
