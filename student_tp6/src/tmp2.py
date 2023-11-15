#%%
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

#%%

logging.basicConfig(level=logging.INFO)

FILE = "../data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


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
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=128
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
#%%
#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
# implémentez l'encodeur-décodeur. Utilisez dans les deux cas des GRUs et les architectures suivantes :
# encodeur : un embedding du vocabulaire d'origine puis un GRU
# décodeur : un embedding du vocabulaire de destination, puis un GRU suivi d'un réseau linéaire pour le décodage de l'état latent 
# (et un softmax pour terminer) Dans le décodeur, vous aurez besoin d'une méthode generate(hidden,lenseq=None) 
# qui à partir d'un état caché hidden (et du token SOS en entrée) produit une séquence jusqu'à ce que la longueur lenseq 
# soit atteinte ou jusqu'à ce que le token EOS soit engendré.

class Encoder(nn.Module):
    """
    Encoder 
    """
    def __init__(self, input_size, hidden_size) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.gru(x)

        return x, hidden   

# %%
enc = Encoder(len(vocEng), 256).to(device)
x, x_len, y, y_len = next(iter(train_loader))
x, y = x.to(device), y.to(device)
print(x.shape)
# %%
enc_out, enc_hid = enc(x)
print(enc_out.shape, enc_hid.shape)
# %%
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax  =nn.Softmax()

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)
        x = self.softmax(x)
        return x, hidden
    
    def generate(self, hidden, lenseq=None):
        """
        hidden: hidden state of the decoder
        lenseq: length of the sequence to generate
        """
        if lenseq is None:
            lenseq = MAX_LEN
        batch_size = hidden.shape[1]
        x = torch.tensor([Vocabulary.SOS]*batch_size).to(device)
        x = x.unsqueeze(0)
        res = []
        for i in range(lenseq):
            x, hidden = self.forward(x, hidden)
            x = torch.argmax(x, dim=2)
            res.append(x)
        return torch.cat(res, dim=0)
    
# %%
decoder = Decoder(len(vocFra), 256).to(device)

# %%
dec_out, dec_hid = decoder(x, enc_hid)
# %%
dec_out.shape, dec_hid.shape
# %%
x.shape, y.shape
# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(enc.parameters()) + list(decoder.parameters()), lr=0.001)

def train(enc, decoder, criterion, optimizer, train_loader, test_loader, epochs=10):
    for epoch in range(epochs):
        enc.train()
        decoder.train()
        for x, _, y, _ in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            enc_out, enc_hid = enc(x)
            dec_hid = enc_hid
            dec_input = torch.tensor([vocFra.SOS]*y.shape[0]).unsqueeze(0).to(device)
            dec_output = torch.zeros(y.shape[0], y.shape[1], len(vocFra)).to(device)
            for i in range(y.shape[1]):
                dec_out, dec_hid = decoder(dec_input, dec_hid)
                dec_output[:, i, :] = dec_out.squeeze(0)
                dec_input = y[:, i].unsqueeze(0)
            loss = criterion(dec_output.view(-1, len(vocFra)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dec_hid = enc_hid
            dec_input = torch.tensor([vocFra.SOS]*y.shape[0]).unsqueeze(0).to(device)
            dec_output = torch.zeros(y.shape[0], y.shape[1], len(vocFra)).to(device)
            for i in range(y.shape[1]):
                dec_out, dec_hid = decoder(dec_input, dec_hid)
                dec_output[:, i, :] = dec_out.squeeze(1)
                dec_input = y[:, i].unsqueeze(0).unsqueeze(0)
            loss = criterion(dec_output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - loss: {loss}")
        enc.eval()
        decoder.eval()
        with torch.no_grad():
            for x, _, y, _ in tqdm(test_loader):
                x, y = x.to(device), y.to(device)
                enc_out, enc_hid = enc(x)
                dec_hid = enc_hid
                dec_input = torch.tensor([vocFra.SOS]*y.shape[0]).unsqueeze(0).to(device)
                dec_output = torch.zeros(y.shape[0], y.shape[1], len(vocFra)).to(device)
                for i in range(y.shape[1]):
                    dec_out, dec_hid = decoder(dec_input, dec_hid)
                    dec_output[:, i, :] = dec_out.squeeze(1)
                    dec_input = y[:, i].unsqueeze(0).unsqueeze(0)
                loss = criterion(dec_output, y)
        print(f"Epoch {epoch+1}/{epochs} - test loss: {loss}")
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/test", loss, epoch)

#%%
train(enc, decoder, criterion, optimizer, train_loader, test_loader, epochs=10)
# %%
def translate(enc, decoder, sentence):
    enc.eval()
    decoder.eval()
    with torch.no_grad():
        sentence = torch.tensor([vocEng.get(w) for w in sentence.split(" ")]).to(device)
        enc_out, enc_hid = enc(sentence.unsqueeze(0))
        res = decoder.generate(enc_hid)
        res = vocFra.getwords(res.squeeze(1).cpu().numpy())
        res = " ".join(res)
        return res
# %%
translate(enc, decoder, "I am a student")
# %%
