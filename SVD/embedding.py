import networkx as nx
import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
import torch
from dataset import split_train_test
from dataset import get_inters_and_id
class node_embedding:
    def __init__(self, interactions_file, G_file, nodes):
        self.G_file = G_file
        self.inters = interactions_file
        self.nodes = list(nodes)

    def get_node_embeds(self):
        G_train = nx.read_adjlist(self.G_file)
        adjacency_matrix = nx.adjacency_matrix(G_train, self.nodes)
        adjacency_matrix = adjacency_matrix.astype(float)
        U, Sigma, VT = svds(adjacency_matrix, k=128)
        Sigma = np.diag(Sigma)
        W = np.matmul(U, np.sqrt(Sigma))
        C = np.matmul(VT.T, np.sqrt(Sigma))
        embeddings = W + C
        rna_embedding = embeddings[0:935, :]
        pro_embedding = embeddings[935:, :]
        return rna_embedding, pro_embedding

class edge_embedding:
    def __init__(self, rna_embeds, pro_embeds, train_num, test_num, testing_pos_edges, labels, interactions_file):
        self.rna_embeds = torch.FloatTensor(rna_embeds)
        self.pro_embeds = torch.FloatTensor(pro_embeds)
        self.train_num = train_num
        self.test_num = test_num
        self.testing_pos_edges = testing_pos_edges
        self.labels = labels
        self.inters = interactions_file

    def get_pos_neg_index(self):
        inters_id = get_inters_and_id(self.inters)
        _, rna_id, pro_id = inters_id.get_iters_rna_id_pro_id()
        pos = np.where(self.labels == 1)[0]
        neg = np.where(self.labels != 1)[0]
        np.random.seed(2022)
        np.random.shuffle(neg)
        neg = neg[0:pos.shape[0] * 1]
        neg_index = list(neg)
        pos_index = list(pos)

        pos_test_index = []
        for item in self.testing_pos_edges:
            l_id, p_id = item
            rna_index=np.where(rna_id==l_id)[0]
            pro_index=np.where(pro_id==p_id)[0]
            pos_test_index.append(int(rna_index*len(pro_id)+pro_index))
            pos_index.remove(rna_index*len(pro_id)+pro_index)
        pos_train_index = pos_index
        neg_train_index = neg_index[0:self.train_num]
        neg_test_index = neg_index[self.train_num:]
        return pos_train_index, pos_test_index, neg_train_index, neg_test_index

    def get_edge_embeds(self):
        pos_train_index, pos_test_index, neg_train_index, neg_test_index = self.get_pos_neg_index()
        a = self.rna_embeds.unsqueeze(1).expand(self.rna_embeds.shape[0], self.pro_embeds.shape[0], self.rna_embeds.shape[1]).reshape((-1, self.rna_embeds.shape[-1]))
        p = self.pro_embeds.unsqueeze(0).expand(self.rna_embeds.shape[0], self.pro_embeds.shape[0], self.pro_embeds.shape[1]).reshape((-1, self.pro_embeds.shape[-1]))
        pos_train_feas = torch.cat([a[pos_train_index], p[pos_train_index]], dim=1)
        pos_test_feas = torch.cat([a[pos_test_index], p[pos_test_index]], dim=1)
        neg_train_feas = torch.cat([a[neg_train_index], p[neg_train_index]], dim=1)
        neg_test_feas = torch.cat([a[neg_test_index], p[neg_test_index]], dim=1)
        return np.array(pos_train_feas.data), np.array(pos_test_feas.data), np.array(neg_train_feas.data), np.array(neg_test_feas.data)