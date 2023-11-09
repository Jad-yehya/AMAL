import torch
import torch.nn as nn
from torch.utils.data import Dataset

# mps device
# device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
device = "cpu"


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
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
        # hs = [torch.zeros(x.size(1), self.latent).to(device)]
        # for i in range(x.size(0)):
        #     hs.append(self.one_step(x[i], hs[-1]))
        # return torch.stack(hs[1:])
        h = torch.zeros(x.size(0), self.latent).to(device)
        h_final = torch.zeros(h.size(0), x.size(1), self.latent).to(device)
        # Batch first
        for i in range(x.size(1)):
            h = self.one_step(x[:, i, :], h)
            h_final[:, i, :] = h
        return h_final



    def decode(self, h):
        """
        Décode un état caché
        :param h: batch*latent
        :return : batch*output
        """
        return self.Wd(h)


class SampleMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else \
            torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes * self.nb_days * (self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots - self.length) * self.nb_days)
        i = i % ((self.nb_timeslots - self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot + self.length), station], station


class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else \
            torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days * (self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot + self.length - 1)], self.data[day,
                                                                      (timeslot + 1):(timeslot + self.length)]
