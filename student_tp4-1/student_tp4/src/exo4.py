import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ''  ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))
batch_size = 32
PATH = "data/"


def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip() + "." for p in full_text.split(".") if len(p) > 0]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long), t])
        return t[:-1], t[1:]


#  TODO:

data_trump = DataLoader(TrumpDataset(open(PATH + "trump_full_speech.txt", "rb").read().decode(), maxlen=1000),
                        batch_size=batch_size, shuffle=True)

# L'objectif est de faire un modèle de langage qui génère des phrases de Trump. On se limitera à des phrases de
# taille fixe (MAX_LEN).

# Modele déjà implémenté dans utils.py (RNN)

# Pour générer une phrase, on commence par générer un état caché initial aléatoire, puis on génère un caractère
# à la fois en utilisant l'état caché précédent et le caractère précédent. On s'arrête lorsque le caractère généré
# est un point.

# Entrainement du modèle de langage
# input: Any, latent: Any, output: Any

model = RNN(len(LETTRES) + 1, 100, len(LETTRES) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x, y = next(iter(data_trump))
print(code2string(x[0]))

# On veut prédire la séquence à t+1 à partir de la séquence jusqu'à t

# Génération de texte
# On commence par générer un état caché initial aléatoire, puis on génère un caractère à la fois en utilisant
# l'état caché précédent et le caractère précédent. On s'arrête lorsque le caractère généré est un point.


def generate(model, maxlen=100):
    """ Génère une phrase à partir d'un modèle """
    model.eval()
    with torch.no_grad():
        h = torch.randn(1, model.latent).to(device)
        x = torch.zeros(1, len(LETTRES) + 1).to(device)
        res = ''
        for i in range(maxlen):
            x = model.one_step(x, h)
            h = x
            x = torch.argmax(x, dim=1)
            res += id2lettre[x.item()]
            if x == 0:
                break
            print(id2lettre[x.item()], end='')
        print()
        return res

# On peut maintenant générer des phrases à partir du modèle non entrainé


generate(model)






