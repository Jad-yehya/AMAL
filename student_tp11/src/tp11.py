# %%
import logging
import math
import random
import time

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import construct_graph, random_walk

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
##  TODO:


# 1. Triplet loss et topologie locale
class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes2id, id2nodes):
        self.graph = graph
        self.nodes2id = nodes2id
        self.id2nodes = id2nodes

    def __len__(self):
        return len(self.graph.nodes())

    def __getitem__(self, idx):
        idx = self.id2nodes[idx]
        anchor = idx
        pos = self.graph.neighbors(idx)
        neg = random.choice(list(self.graph.nodes()))
        while neg in pos:
            neg = random.choice(list(self.graph.nodes()))

        pos = random.choice(list(self.graph.neighbors(idx)))
        return self.nodes2id[anchor], self.nodes2id[pos], self.nodes2id[neg]


# 2. Node2Vec
class Node2VecDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes2id, id2nodes, walks):
        self.graph = graph
        self.nodes2id = nodes2id
        self.id2nodes = id2nodes
        self.walks = walks

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        return torch.Tensor([self.nodes2id[node] for node in self.walks[idx]])

    def collate_fn(self, batch):
        return batch

    def get_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    PATH = "../data/ml-latest-small/"
    logging.info("Constructing graph")

    method = 1

    movies_graph, movies = construct_graph(PATH + "movies.csv", PATH + "ratings.csv")

    if method == 1:
        logging.info("Sampling walks")
        walks = random_walk(movies_graph, 5, 10, 1, 1)
    nodes2id = dict(zip(movies_graph.nodes(), range(len(movies_graph.nodes()))))
    id2nodes = list(movies_graph.nodes())
    id2title = [movies[movies.movieId == idx].iloc[0].title for idx in id2nodes]
    ##  TODO:
    td = TripleDataset(movies_graph, nodes2id, id2nodes)
    print(td.__getitem__(1331))

    if method == 0:  #  Triplet Loss
        model = nn.Embedding(len(movies_graph.nodes()), 1000)
        model.to(device)

        dataloader = DataLoader(td, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.TripletMarginLoss()

        writer = SummaryWriter()

        for epoch in range(10):
            for anchor, pos, neg in tqdm(dataloader):
                optimizer.zero_grad()
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                anchor = model(anchor)
                pos = model(pos)
                neg = model(neg)
                loss = criterion(anchor, pos, neg)
                loss.backward()
                optimizer.step()
            writer.add_scalar("loss", loss, epoch)
            writer.add_embedding(model.weight, global_step=epoch)
            writer.flush()

        # Visualisation des embeddings avec t-sne
        embeddings = model.weight.detach().cpu().numpy()

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)

        import matplotlib.pyplot as plt

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.show()

    elif method == 1:  #  Node2Vec
        model = nn.Embedding(len(movies_graph.nodes()), 1000)
        model.to(device)

        dataloader = Node2VecDataset(
            movies_graph, nodes2id, id2nodes, walks
        ).get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        writer = SummaryWriter()

        for epoch in range(10):
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                batch = batch.to(device)
                batch = model(batch)
                loss = criterion(batch, batch[:, 0])
                loss.backward()
                optimizer.step()
            writer.add_scalar("Node2vec Loss", loss, epoch)
            writer.add_embedding(model.weight, global_step=epoch)
            writer.flush()

        # Visualisation des embeddings avec t-sne
        embeddings = model.weight.detach().cpu().numpy()

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(embeddings)

        import matplotlib.pyplot as plt

        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.show()


# %%
