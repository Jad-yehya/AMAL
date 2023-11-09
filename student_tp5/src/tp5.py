import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  TODO:


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: code du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle,
    #   la CrossEntropy qui ne prend pas en compte les caractères de padding.
    loss = CrossEntropyLoss(
        output, target, reduce="none"
    )  # Calcul de la ll pour chaque exemple
    mask = (target != padcar).float()  # Masque des exemples non paddés
    return (
        loss * mask
    ).sum() / mask.sum()  # Moyenne pondérée par le nombre d'exemples non paddés


class RNN(nn.Module):
    # TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, input, latent, output) -> None:
        super(RNN, self).__init__()
        self.input = input
        self.latent = latent
        self.output = output
        self.Wi = nn.Linear(input, latent)
        self.Wh = nn.Linear(latent, latent)
        self.Wd = nn.Linear(latent, output)
        # Initialisation des poids avec kaiming (meilleur stabilité)
        # nn.init.kaiming_normal_(self.Wi.weight, nonlinearity='tanh')
        # nn.init.kaiming_normal_(self.Wh.weight, nonlinearity='tanh')
        # nn.init.kaiming_normal_(self.Wd.weight, nonlinearity='tanh')

    def one_step(self, x, h):
        """
        Traite un pas de temps
        :param x: seq*batch*dim
        :param h: batch*latent
        """
        return torch.tanh(self.Wi(x) + self.Wh(h))

    def forward(self, x):
        """
        Traite une séquence complète
        :param x: seq*batch*dim
        :return : séquence d'états cachés seq*batch*latent
        """
        hs = [torch.zeros(x.size(1), self.latent).to(device)]
        for i in range(x.size(0)):
            hs.append(self.one_step(x[i], hs[-1]))
        return torch.stack(hs[1:])

    def decode(self, h):
        """
        Décode un état caché
        :param h: batch*latent
        :return : batch*output
        """
        return self.Wd(h)


class LSTM(RNN):
    # TODO:  Implémenter un LSTM
    def __init__(self, input, latent, output, c):
        super(LSTM, self).__init__(input, latent, output)
        self.Wf = nn.Linear(input, latent)
        self.Wo = nn.Linear(input, latent)
        self.Wi = nn.Linear(input, latent)
        self.Wc = nn.Linear(input, latent)
        self.c = c # mémorise interne

    def one_step(self, x, h):
        f = torch.sigmoid(self.Wf(x) + self.Wf(h))
        i = torch.sigmoid(self.Wi(x) + self.Wi(h))
        self.c = f * self.c + i * torch.tanh(self.Wc(x) + self.Wc(h))
        o = torch.sigmoid(self.Wo(x) + self.Wo(h))
        return o * torch.tanh(self.c)

    def forward(self, x):
        hs = [torch.zeros(x.size(1), self.latent).to(device)]
        for i in range(x.size(0)):
            hs.append(self.one_step(x[i], hs[-1]))
        return torch.stack(hs[1:])


class GRU(nn.Module):
    # TODO:  Implémenter un GRU
    def __init__(self, input, latent, output):
        super(GRU, self).__init__()
        self.input = input
        self.latent = latent
        self.output = output
        self.Wz = nn.Linear(input, latent, bias=False)
        self.Wr = nn.Linear(input, latent, bias=False)

        self.W = nn.Linear(input, latent)

    def one_step(self, x, h):
        """
        Traite un pas de temps
        :param x: seq*batch*dim
        :param h: batch*latent
        """
        z = torch.sigmoid(self.Wz(x) + self.Wz(h))
        r = torch.sigmoid(self.Wr(x) + self.Wr(h))
        h_tilde = torch.tanh(self.W(x) + self.W(r * h))
        return (1 - z) * h + z * h_tilde

    def forward(self, x):
        """
        Traite une séquence complète
        :param x: seq*batch*dim
        :return : séquence d'états cachés seq*batch*latent
        """
        hs = [torch.zeros(x.size(1), self.latent).to(device)]
        for i in range(x.size(0)):
            hs.append(self.one_step(x[i], hs[-1]))
        return torch.stack(hs[1:])


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
