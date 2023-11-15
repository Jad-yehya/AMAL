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


logging.basicConfig(level=logging.INFO)

FILE = "data/en-fra.txt"

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
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
# implémentez l'encodeur-décodeur. Utilisez dans les deux cas des GRUs et les architectures suivantes :
# encodeur : un embedding du vocabulaire d'origine puis un GRU
# décodeur : un embedding du vocabulaire de destination, puis un GRU suivi d'un réseau linéaire pour le 
# décodage de l'état latent (et un softmax pour terminer) Dans le décodeur, vous aurez besoin d'une 
# méthode generate(hidden,lenseq=None) qui à partir d'un état caché hidden (et du token SOS en entrée) 
# produit une séquence jusqu'à ce que la longueur lenseq soit atteinte ou jusqu'à ce que le token EOS soit engendré.

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size=256):
        super(Encoder, self).__init__()
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, -1, self.embedding_size)
        output, hidden = self.gru(x)
        return output, hidden

    def initHidden(self): 
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size=256):
        super(Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        """
        Prend l'output de l'encodeur et le hidden state et renvoie la prédiction et le nouveau hidden state
        """
        batch_size = x.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(Vocabulary.SOS)
        decoder_output = []
        for i in range(x.size(1)):
            decoder_input = self.embedding(decoder_input)
            output, hidden = self.gru(decoder_input, hidden)
            output = self.linear(output)
            output = self.softmax(output)
            decoder_output.append(output)
            decoder_input = output.argmax(dim=2)

        decoder_output = torch.cat(decoder_output, dim=1)
        return decoder_output, hidden


    def forward_step(self, x, hidden):
        """
        Prend en entrée le dernier token généré et le hidden state et renvoie la prédiction et le nouveau hidden state
        """
        x = self.embedding(x).view(1, -1, self.embedding_size)
        output, hidden = self.gru(x, hidden)
        x = self.linear(output)
        x = self.softmax(x)
        return x, hidden

    def generate(self,hidden,lenseq=None):
        x = torch.tensor([[Vocabulary.SOS]], device=device)
        res = []
        if lenseq is None:
            lenseq = 100

        for i in range(lenseq):
            x, hidden = self.forward(x, hidden)
            topv, topi = x.topk(1)
            res.append(topi.item())
            if topi.item() == Vocabulary.EOS or (lenseq is not None and len(res) >= lenseq):
                break
            x = topi.squeeze().detach()
        return res,hidden

    def initHidden(self): 
        return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embedding_size=256):
        super(EncoderDecoder, self).__init__()
        self.Encoder = Encoder(input_size, hidden_size, embedding_size)
        self.Decoder = Decoder(output_size, hidden_size, embedding_size)
        self.output_size = output_size

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        x_len = x.shape[0]
        y_len = y.shape[0]
        batch_size = x.shape[1]
        vocab_size = self.output_size

        outputs = torch.zeros(y_len, batch_size, vocab_size).to(device)
        hidden = self.Encoder.initHidden()
        for i in range(x_len):
            _, hidden = self.Encoder(x[i], hidden)

        input = y[0]
        for i in range(1, y_len):
            output, hidden = self.Decoder(input, hidden)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (y[i] if teacher_force else top1)
        return outputs

    def generate(self, x, lenseq=None):
        x_len = x.shape[0]
        batch_size = x.shape[1]
        hidden = self.Encoder.initHidden()
        for i in range(x_len):
            _, hidden = self.Encoder(x[i], hidden)

        res = []
        for i in range(batch_size):
            res.append(self.Decoder.generate(hidden[:, i, :],lenseq)[0])
        return res


def train(model, train_loader, dev_loader, loss_function, optimizer, writer, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, x_len, y, y_len = batch
            
            print("x :", x.shape)
            print("x_len :", x_len.shape)
            print("y :", y.shape)
            print("y_len :", y_len.shape)

            x = x.permute(1, 0).to(device)
            y = y.permute(1, 0).to(device)

            model.zero_grad()
            output = model(x, y)
            loss = loss_function(output.view(-1, model.output_size), y[1:].view(-1))
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), epoch)
        logging.info("Epoch %d: loss %f", epoch, loss.item())
        evaluate(model, dev_loader, loss_function, writer, epoch)


def evaluate(model, dev_loader, loss_function, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dev_loader:
            x, x_len, y, y_len = batch

            x = x.to(device)
            y = y.to(device)

            output = model(x, y)
            loss = loss_function(output.view(-1, model.output_size), y[1:].view(-1))
            total_loss += loss.item()
    writer.add_scalar("Loss/dev", total_loss, epoch)
    logging.info("Evaluation: loss %f", total_loss)

def test(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        total = 0
        correct = 0
        for batch in test_loader:
            x, y = batch
            x = x.to_device()
            y = y.to_device()

            output = model(x, y)
            loss = loss_function(output.view(-1, model.output_size), y[1:].view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            predicted = predicted.tolist()
            y = y.tolist()
            for i in range(len(predicted)):
                if y[i] != Vocabulary.PAD:
                    total += 1
                    if predicted[i] == y[i]:
                        correct += 1

    writer.add_scalar("Loss/test", total_loss)

    logging.info("Evaluation: loss %f", total_loss)
    logging.info("Accuracy: %f", correct / total)

def generate(model, test_loader):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to_device()
            y = y.to_device()

            output = model.generate(x)
            print(output)
            print(vocFra.getwords(output))
            print(vocFra.getwords(y.tolist()))
            break

if __name__ == "__main__":
    input_size = len(vocEng)
    output_size = len(vocFra)
    hidden_size = 256
    embedding_size = 256
    model = EncoderDecoder(input_size, output_size, hidden_size, embedding_size).to(device)
    loss_function = nn.NLLLoss(ignore_index=Vocabulary.PAD)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, train_loader, test_loader, loss_function, optimizer, writer, epochs=10)
    test(model, test_loader)
    generate(model, test_loader)

    # encoder = Encoder(input_size, hidden_size, embedding_size).to(device)
    # decoder = Decoder(output_size, hidden_size, embedding_size).to(device)

    # batch = next(iter(train_loader))
    # x, x_len, y, y_len = batch

    # x = x.permute(1, 0).to(device)
    # y = y.permute(1, 0).to(device)

    # hidden = encoder.initHidden()
    # for i in range(x.shape[0]):
    #     _, hidden = encoder(x[i], hidden)

    # decoder_input = torch.tensor([[Vocabulary.SOS]], device=device)

    # outs = []
    # for i in range(y.shape[0]):
    #     output, hidden = decoder(decoder_input, hidden)
    #     topv, topi = output.topk(1)
    #     outs.append(topi.item())
    #     decoder_input = topi.squeeze().detach()
    #     if topi.item() == Vocabulary.EOS:
    #         break

    # translation = [vocFra.getword(i) for i in outs]
    # print(translation)
writer.close()




