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


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("tp6-traduction.log"),
                              logging.StreamHandler()])


FILE = "./data/en-fra.txt"

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


class Encoder(nn.Module):
    """
    Encoder module of the seq2seq model.
    Args:
        input_size (int): The size of the input vocabulary.
        hidden_size (int): The size of the hidden state of the GRU.
    """
    def __init__(self, input_size, hidden_size) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.gru(x)
        return x, hidden


class Decoder(nn.Module):
    """
    Deocder : takes encoder outputs and hidden state as input and generates output sequence.

    Args:
        hidden_size (int): The number of expected features in the input hidden state.
        output_size (int): The size of the output vocabulary.

    """
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


def train(model, optimizer, loss_function, train_loader, val_loader, epochs=10, teacher_forcing=0.5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, x_len, y, y_len in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            encoder_outputs, encoder_hidden = model.encoder(x)
            if random.random() < teacher_forcing:
                # Teacher forcing case, we use the target tensor
                decoder_outputs, _ = model.decoder(encoder_outputs, encoder_hidden, y)
            else:
                # Without teacher forcing
                decoder_outputs, _ = model.decoder(encoder_outputs, encoder_hidden)
            
            loss = loss_function(decoder_outputs.view(-1, len(vocFra)), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # Train accuracy
        _, topi = decoder_outputs.topk(1)
        topi = topi.squeeze().detach().view(-1, y.shape[1])
        train_accuracy = (topi == y).sum().item() / (y.size(0) * y.size(1))
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        logging.info(f"Epoch {epoch} : train loss {train_loss}, train accuracy {train_accuracy}")

        # Validation
        model.eval()
        val_loss = 0
        past_acc = 0
        for x, x_len, y, y_len in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            encoder_outputs, encoder_hidden = model.encoder(x)
            decoder_outputs, _ = model.decoder(encoder_outputs, encoder_hidden, y)
            loss = loss_function(decoder_outputs.view(-1, len(vocFra)), y.view(-1))
            val_loss += loss.item()
        val_loss /= len(val_loader)
        # Validation accuracy
        _, topi = decoder_outputs.topk(1)
        topi = topi.squeeze().detach().view(-1, y.shape[1])
        correct = (topi == y).sum().item() / (y.size(0) * y.size(1))
        writer.add_scalar('Accuracy/val', correct, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        logging.info(f"Epoch {epoch} : val loss {val_loss}, val accuracy {correct}")
        
        if correct > past_acc:
            past_acc = correct
            torch.save(model.state_dict(), f"model-best.pt")

if __name__ == "__main__":
    epochs = 15
    hidden_size = 1024
    learning_rate = 3e-3
    teacher_forcing = 1

    encoder = Encoder(len(vocEng), hidden_size).to(device)
    decoder = Decoder(hidden_size, len(vocFra)).to(device)
    model = nn.ModuleDict({
        "encoder": encoder,
        "decoder": decoder
    })

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)

    logging.info("Starting training of the translation model")
    logging.info(f"Using device {device}")
    logging.info(f"Number of epochs {epochs}, hidden size {hidden_size}, learning rate {learning_rate}, teacher forcing {teacher_forcing}")

    train(model, optimizer, loss_function, train_loader, test_loader, epochs, teacher_forcing)

    logging.info("Training finished")