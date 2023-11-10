import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time

logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Format de sortie décrit dans
# https://pypi.org/project/conllu/

PAD = 0
OOV = 1

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation :

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = OOV
    PAD = PAD

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov = oov
        self.id2word = ["PAD"]
        self.word2id = {"PAD": Vocabulary.PAD}
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


class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s],
                                   [tags.get(token["upostag"], adding) for token in s]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)

logging.info("Vocabulary size: %d", len(words))

BATCH_SIZE = 100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)
# Implémentez un modèle seq2seq pour le tagging et la boucle d'apprentissage en utilisant le module LSTM de Pytorch
# et le padding de séquences. Pour tenir compte des mots OOV, vous pouvez utiliser le token "OOVID" (déjà défini
# dans la classe Vocabulary).
class Tagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(Tagger, self).__init__()
        # We have batch first
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # logging.info("Embeds shape: %s", embeds.shape)
        lstm_out, _ = self.lstm(embeds)
        # logging.info("LSTM out shape: %s", lstm_out.shape)
        tag_space = self.hidden2tag(lstm_out)
        # logging.info("Tag space shape: %s", tag_space.shape)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        # logging.info("Tag scores shape: %s", tag_scores.shape)
        return tag_scores


def train(model, train_loader, dev_loader, loss_function, optimizer, writer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Putting batch first
            x, y = batch
            x = x.permute(1, 0).to(device)
            y = y.permute(1, 0).to(device)
            model.zero_grad()
            output = model(x)
            output = output.permute(0, 2, 1)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar("Loss/train", total_loss, epoch)
        logging.info("Epoch %d: loss %f", epoch, total_loss)
        evaluate(model, dev_loader, loss_function, writer, epoch)


def evaluate(model, dev_loader, loss_function, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dev_loader:
            x, y = batch
            x = x.permute(1, 0).to(device)
            y = y.permute(1, 0).to(device)
            output = model(x)
            output = output.permute(0, 2, 1)
            loss = loss_function(output, y)
            total_loss += loss.item()
    writer.add_scalar("Loss/dev", total_loss, epoch)
    logging.info("Evaluation: loss %f", total_loss)


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        # Computing the accuracy
        total = 0
        correct = 0
        for batch in test_loader:
            x, y = batch
            x = x.permute(1, 0).to(device)
            y = y.permute(1, 0).to(device)
            output = model(x)
            output = output.permute(0, 2, 1)

            # Getting the best tag for each word
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.tolist()
            y = y.tolist()

            # Computing the accuracy only for non PAD words
            for i in range(len(predicted)):
                for j in range(len(predicted[i])):
                    if y[i][j] != 0:
                        total += 1
                        if predicted[i][j] == y[i][j]:
                            correct += 1

        logging.info("Accuracy: %f", correct / total)


def main():
    logging.info("Starting...")
    logging.info("Using device %s", device)
    writer = SummaryWriter()
    model = Tagger(len(words), len(tags), 128, 128)
    model.to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, dev_loader, loss_function, optimizer, writer, epochs=20)
    test(model, test_loader)
    writer.close()


if __name__ == "__main__":
    main()
