from utils import RNN, device, ForecastMetroDataset

from torch.utils.data import DataLoader
import torch
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
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles
# Modèle de séquences multivariées
# L’objectif de cette partie est de faire de la prédiction de séries temporelles :
# à partir d’une séquence de flux de longueur t pour l’ensemble des stations du jeu de données,
# on veut prédire le flux à t + 1, t + 2, ...
# Vous entraînerez un RNN commun à toutes les stations qui prend une série dans Rn×2 et prédit une série dans Rn×2.

# Que doit-on changer au modèle précédent ? Quel coût est dans ce cas plus adapté que la cross-entropie ?
# Faire les expériences en faisant varier l’horizon de prédiction (à t+2, etc.) et la longueur des séquences en entrée.
# Vous pouvez comme précédemment considérer d’abord que le flux entrant, puis le flux entrant et sortant.

# Dans ce contexte de réseau many-to-many, la supervision peut se faire à chaque étape de la séquence sans attendre
# la fin de la séquence. La rétro-propagation n’est faîte qu’une fois que toute la séquence a été vue,
# mais à un instant t, le gradient prend en compte l’erreur à ce moment (en fonction de la supervision du décodage)
# mais également l’erreur des pas de temps d'après qui cumulée.

model = RNN(DIM_INPUT, 20, DIM_INPUT).to(device)
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

train_accu = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES)
test_accu = torchmetrics.classification.Accuracy(task="multiclass", num_classes=CLASSES)

if False:

    # Overfitting sur un batch
    X, Y = next(iter(data_train))
    X = X.to(device)
    Y = Y.to(device)

    print(X.shape)
    print(Y.shape) # Y.shape = X.shape = batch*seq-1*latent*dim

    # On veut prédire la séquence à t+1 à partir de la séquence jusqu'à t

    for i in range(100):
        losses = 0
        for c in range(CLASSES):
            x_c, y_c = X[:, :, c, :], Y[:, :, c, :]
            h_out = model(x_c)  # h_out.shape = batch*seq-1*latent
            pred = model.decode(h_out)  # pred.shape = batch*seq-1*2
            loss = criterion(pred, y_c)
            losses += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Epoch %d : loss = %.3f" % (i, losses))

    exit()

# Maintenant sur tout le dataset

train_loss = []
val_loss = []
for i in range(100):
    for X, Y in data_train:
        X = X.to(device)
        Y = Y.to(device)
        losses = 0
        for c in range(CLASSES):
            x_c, y_c = X[:, :, c, :], Y[:, :, c, :]
            h_out = model(x_c)
            pred = model.decode(h_out)
            loss = criterion(pred, y_c)
            losses += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_loss.append(losses)

    with torch.no_grad():
        for X, Y in data_test:
            X = X.to(device)
            Y = Y.to(device)
            losses = 0
            for c in range(CLASSES):
                x_c, y_c = X[:, :, c, :], Y[:, :, c, :]
                h_out = model(x_c)
                pred = model.decode(h_out)
                loss = criterion(pred, y_c)
                losses += loss.item()
            val_loss.append(losses)

    print("Epoch %d : train_loss = %.3f, val_loss = %.3f" % (i, train_loss[-1], val_loss[-1]))
