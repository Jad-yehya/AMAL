import numpy as np

from utils import RNN, device, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
import torchmetrics

# Nombre de stations utilisé
CLASSES = 10
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32

PATH = "data/"

matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#%%
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
model = RNN(DIM_INPUT, 10, CLASSES).to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-2)

train_accu = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES)
test_accu = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES)

#%%
if False:
    # Overfitting sur un batch
    X, Y = next(iter(data_train))
    X = X.to(device)
    Y = Y.to(device)

    print(X.shape)
    print(Y.shape)

    for i in range(100):
        h_out = model(X)  # Toutes les sorties cachées - seq*batch*latent
        pred = model.decode(h_out[:, -1])  # Dernière sortie cachée décodée (many-to-one) - batch*output
        loss = criterion(pred, Y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("Epoch %d : loss = %.3f" % (i, loss.item()))

    exit()

# Maintenant sur tout le dataset

train_loss = []
val_loss = []
for i in range(100):
    for X, Y in data_train:
        X = X.to(device)
        Y = Y.to(device)

        optim.zero_grad()
        h_out = model(X)  # Toutes les sorties cachées - seq*batch*latent
        # On calcule la loss avec les logits
        pred = model.decode(h_out[:, -1])  # Dernière sortie cachée décodée (many-to-one) - batch*output
        loss = criterion(pred, Y)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        # Accuracy
        # pred = torch.nn.functional.softmax(pred, dim=1)
        # tracc = (pred.argmax(dim=1) == Y).float().mean()
        train_accu(pred.argmax(dim=1), Y)

    with torch.no_grad():
        for X, Y in data_test:
            X = X.to(device)
            Y = Y.to(device)
            h_out = model(X)
            pred = model.decode(h_out[:, -1])
            loss = criterion(pred, Y)
            val_loss.append(loss.item())
            # Accuracy
            # pred = torch.nn.functional.softmax(pred, dim=1)
            # tsacc = (pred.argmax(dim=1) == Y).float().mean()
            test_accu(pred.argmax(dim=1), Y)

    # print("Epoch %d : train loss = %.3f, val loss = %.3f" % (i, np.mean(train_loss), np.mean(val_loss)),
    #       "Accuracy : train = %.3f, test = %.3f" % (tracc, tsacc))

    print("Epoch %d : train loss = %.3f, val loss = %.3f" % (i, np.mean(train_loss), np.mean(val_loss)),
          "Accuracy : train = %.3f, test = %.3f" % (train_accu.compute(), test_accu.compute()))
    train_accu.reset()
    test_accu.reset()


# Save the stats
import matplotlib.pyplot as plt

plt.plot(train_loss, label="train")
plt.plot(val_loss, label="test")
plt.title("Loss")
plt.legend()
plt.savefig("train_test_loss.png")
plt.show()

