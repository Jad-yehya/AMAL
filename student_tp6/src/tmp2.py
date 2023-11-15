# %%
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import random

import time
import re
# from torch.utils.tensorboard import SummaryWriter

# %%
# Make logging to a file and to the console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[
                        logging.FileHandler("tmp2.log"),
                        logging.StreamHandler()
                    ]
)

# %%

FILE = "./data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())


def normalize(s):
    return re.sub(' +', ' ', "".join(c if c in string.ascii_letters else " "
                                     for c in unicodedata.normalize('NFD', s.lower().strip())
                                     if c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD,
                        "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TradDataset():
    def __init__(self, data, vocOrig, vocDest, adding=True, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1:
                continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len:
                continue
            self.sentences.append(
                (torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),
                 torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS]))
                 )

    def __len__(self): return len(self.sentences)
    def __getitem__(self, i): return self.sentences[i]


def collate_fn(batch):
    orig, dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig), o_len, pad_sequence(dest), d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN = 128
BATCH_SIZE = 100

datatrain = TradDataset(
    "".join(lines[:idxTrain]), vocEng, vocFra, max_len=MAX_LEN)
datatest = TradDataset(
    "".join(lines[idxTrain:]), vocEng, vocFra, max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn,
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn,
                         batch_size=BATCH_SIZE, shuffle=True)
# %%
#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
# implémentez l'encodeur-décodeur. Utilisez dans les deux cas des GRUs et les architectures suivantes :
# encodeur : un embedding du vocabulaire d'origine puis un GRU
# décodeur : un embedding du vocabulaire de destination, puis un GRU suivi d'un réseau linéaire pour le décodage de l'état latent
# (et un softmax pour terminer) Dans le décodeur, vous aurez besoin d'une méthode generate(hidden,lenseq=None)
# qui à partir d'un état caché hidden (et du token SOS en entrée) produit une séquence jusqu'à ce que la longueur lenseq
# soit atteinte ou jusqu'à ce que le token EOS soit engendré.

hidden_size = 512

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.gru(x)
        return x, hidden
    

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def one_step(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.linear(x)
        return x, hidden

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.empty(1, batch_size,
                                    dtype=torch.long,
                                    device=device).fill_(Vocabulary.SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LEN):
            decoder_output, decoder_hidden = self.one_step(
                decoder_input,
                decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing
                decoder_input = target_tensor[i].view(1, -1)
                if i == target_tensor.size(0)-1:
                    break
            else:
                # Without teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().view(1, -1)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        # decoder_outputs = nn.functional.softmax(decoder_outputs, dim=2)
        # Pas de softmax car CrossEntropyLoss le fait
        return decoder_outputs, decoder_hidden

    def generate(self, hidden, lenseq=None):
        decoder_input = torch.empty(1, 1, dtype=torch.long, device=device)\
            .fill_(Vocabulary.SOS)
        decoder_hidden = hidden
        decoder_outputs = []

        if lenseq is None:
            lenseq = MAX_LEN

        for i in range(MAX_LEN):
            decoder_output, decoder_hidden = self.one_step(
                decoder_input,
                decoder_hidden
            )

            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(1, -1)
            decoder_outputs.append(decoder_output)

            if decoder_input.item() == Vocabulary.EOS:
                break

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        decoder_outputs = nn.functional.softmax(decoder_outputs, dim=2)
        return decoder_outputs, decoder_hidden


# %%
x, x_len, y, y_len = next(iter(train_loader))
x = x.to(device)
y = y.to(device)
print(x.shape, y.shape)
# %%
encoder = Encoder(len(vocEng), hidden_size).to(device)
decoder = Decoder(hidden_size, len(vocFra)).to(device)
# %%
encoder_outputs, encoder_hidden = encoder(x)
print(encoder_outputs.shape, encoder_hidden.shape)
# %%
decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden, y)
print(decoder_outputs.shape, decoder_hidden.shape)

#%%
lr = 3e-3
epochs = 15
#%%
logging.info("Starting training")
logging.info(f"Learning rate : {lr}")
logging.info(f"Batch size : {BATCH_SIZE}")
logging.info(f"Max len : {MAX_LEN}")
logging.info(f"Hidden size : {hidden_size}")
logging.info(f"Epochs : {epochs}")

# %%
# Training

model = nn.Sequential(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for x, x_len, y, y_len in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(x)
        decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden, y)
        loss = criterion(decoder_outputs.view(-1, len(vocFra)), y.view(-1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), epoch)
    logging.info(f"Epoch {epoch} : Train loss : {loss.item()}")
    # Calculating accuracy
    model.eval()
    with torch.no_grad():
        x, x_len, y, y_len = next(iter(test_loader))
        x = x.to(device)
        y = y.to(device)
        encoder_outputs, encoder_hidden = encoder(x)
        decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden, y)
        loss = criterion(decoder_outputs.view(-1, len(vocFra)), y.view(-1))
        writer.add_scalar("Loss/test", loss.item(), epoch)
        logging.info(f"Epoch {epoch} : Test loss : {loss.item()}")
        # Calculating accuracy
        _, topi = decoder_outputs.topk(1)
        topi = topi.squeeze().detach().view(-1, y.shape[1])
        #print(topi.shape)
        #print(y.shape)
        accuracy = (topi == y).sum().item() / (y.shape[0] * y.shape[1])
        logging.info(f"Epoch {epoch} : Test accuracy : {accuracy}")
        writer.add_scalar("Accuracy/test", accuracy, epoch)


# %%
x, x_len, y, y_len = next(iter(test_loader))
x = x.to(device)
y = y.to(device)
print(x[:, 0].shape, y[:, 0].shape)
# %%
print(vocEng.getwords(x[:, 0].tolist()))
print(vocFra.getwords(y[:, 0].tolist()))
# %%
encoder_outputs, encoder_hidden = encoder(x)
decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden, y)
# %%
print(vocFra.getwords(decoder_outputs[:, 0].argmax(dim=1).tolist()))
# %%
