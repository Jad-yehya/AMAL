# %%
from datamaestro import prepare_dataset
import click
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
from pathlib import Path
import os
import logging
logging.basicConfig(level=logging.INFO)


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05


def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


# %%
#  TODO:  Implémenter
ds = prepare_dataset("com.lecun.mnist")

train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

NUM_CLASSES = 10
INPUT_DIM = 28 * 28  # 784


class MnistDataset(Dataset):
    def __init__(self, images, labels, train=False):
        super(MnistDataset, self).__init__()

        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

        if train:
            # 5% des données d’entraînement
            len5 = int(len(images) * TRAIN_RATIO)

            # On veut des données stratifiées, avec le même nombre d’images par classe
            # On récupère les indices des images de chaque classe
            indices = []
            for i in range(10):
                indices.append(torch.where(self.labels == i)[0])

            # On récupère 5% des indices de chaque classe
            indices5 = []
            for i in range(10):
                indices5.append(indices[i][:len5//10])

            # On concatène les indices
            indices5 = torch.cat(indices5)

            # On récupère les images et labels correspondants
            self.images = self.images[indices5]
            self.labels = self.labels[indices5]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


train_dataset = MnistDataset(train_images, train_labels, train=True)
test_dataset = MnistDataset(test_images, test_labels)


train_dataset, val_dataset = random_split(
    train_dataset, [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=300, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=300, shuffle=True)

next(iter(train_loader))[0].shape


class MLP(nn.Module):
    def __init__(self,
                 fan_in=784,
                 fan_hidden=100,
                 fan_out=10,
                 dropout=0,
                 batch_norm=False,
                 layer_norm=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(fan_in, fan_hidden)
        self.fc2 = nn.Linear(fan_hidden, fan_hidden)
        self.fc3 = nn.Linear(fan_hidden, fan_out)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(fan_hidden)
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(fan_hidden)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.layer_norm:
            x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.layer_norm:
            x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(model,
          crtierion,
          optimizer,
          train_loader,
          val_loader,
          device,
          epochs=1000,
          writer=None,
          l1_reg=False,
          l2_reg=False):
    """
    Train the model for a given number of epochs
    :param model:           Model to train
    :param criterion:       Loss function
    :param optimizer:       Optimizer
    :param train_loader:    the train data loader
    :param val_loader:      Validation data loader
    :param device:          Device to train on (cpu or cuda)
    :param epochs:          Number of epochs to train for
    :param writer:          Tensorboard writer
    :param l1_reg:          L1 regularization coefficient, False if no regularization
    :param l2_reg:          L2 regularization coefficient, False if no regularization
    """

    for epoch in tqdm(range(epochs)):
        model.train()
        correct = 0
        total = 0
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Enregistrement des gradients
            model.fc1.weight.retain_grad()
            model.fc2.weight.retain_grad()
            model.fc3.weight.retain_grad()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Adding L1 regularization
            if l1_reg is not False:
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                loss += l1_reg * l1_loss

            # Adding L2 regularization
            if l2_reg is not False:
                l2_loss = 0
                for param in model.parameters():
                    l2_loss += torch.norm(param, 2)
                loss += l2_reg * l2_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (torch.argmax(output, dim=1) == target).sum().item()
            total += target.size(0)

            # poids de chaque couche
            pfc1 = model.fc1.weight
            pfc2 = model.fc2.weight
            pfc3 = model.fc3.weight

            # Calcul de l'entropie sur la sortie
            entropy = -torch.sum(F.softmax(output, dim=1)
                                 * F.log_softmax(output, dim=1), dim=1).mean()

        if writer is not None:
            writer.add_scalar("Loss/train", running_loss /
                              len(train_loader), epoch)
            writer.add_scalar("Accuracy/train", correct/total, epoch)

            if epoch % 50 == 0:

                writer.add_histogram("fc1", pfc1, epoch)
                writer.add_histogram("fc2", pfc2, epoch)
                writer.add_histogram("fc3", pfc3, epoch)

                writer.add_histogram("fc1_grad", pfc1.grad, epoch)
                writer.add_histogram("fc2_grad", pfc2.grad, epoch)
                writer.add_histogram("fc3_grad", pfc3.grad, epoch)

                writer.add_histogram("entropy", entropy, epoch)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                loss = criterion(output, target)

            accuracy = correct/total

            if writer is not None:
                writer.add_scalar("Loss/val", loss.item(), epoch)
                writer.add_scalar("Accuracy/val", correct/total, epoch)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {accuracy}")


def test(model, criterion, test_loader, device, writer=None):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss = criterion(output, target)

        if writer is not None:
            writer.add_scalar("Loss/test", loss.item())
            writer.add_scalar("Accuracy/test", correct/total)

        print(f"Test: {correct/total}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    l1_reg = 0.01
    l2_reg = 0.01
    dropout = 0.2

    print("Training MLP")
    writer = SummaryWriter()
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer)
    test(model, criterion, test_loader, device)
    writer.close()

    # With L1 regularization
    print("L1 regularization")
    writer = SummaryWriter("runs/l1")
    model = MLP().to(device)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer, l1_reg=l1_reg)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()

    # With L2 regularization
    print("L2 regularization")
    writer = SummaryWriter("runs/l2")
    model = MLP().to(device)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer, l2_reg=l2_reg)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()

    # With dropout
    print("Dropout")
    writer = SummaryWriter("runs/dropout")
    model = MLP(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()

    # With batch normalization
    print("Batch norm")
    writer = SummaryWriter("runs/batch_norm")
    model = MLP(batch_norm=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()

    # With layer normalization
    print("Layer norm")
    writer = SummaryWriter("runs/layer_norm")
    model = MLP(layer_norm=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()

    # With batch normalization, dropout and L2 regularization
    print("Batch norm, dropout and L2 regularization")
    writer = SummaryWriter("runs/batch_norm_dropout_l2")
    model = MLP(batch_norm=True, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, train_loader,
          val_loader, device, epochs=1000, writer=writer, l2_reg=l2_reg)
    test(model, criterion, test_loader, device, writer=writer)
    writer.close()
