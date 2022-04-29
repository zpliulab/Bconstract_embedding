import networkx as nx
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import copy
import networkx as nx
import random
import os

class get_inters_and_id:
    def __init__(self, interactions_file):
        self.inters = interactions_file

    def get_iters_rna_id_pro_id(self):
        with open(self.inters) as pid:
            data = pid.readlines()
        rna = []
        pro = []
        inters = []
        for item in data:
            eve_inter = []
            item = item.strip().split('\t')
            eve_inter.append(item[0])
            eve_inter.append(item[1])
            inters.append(eve_inter)
            if item[0] not in rna:
                rna.append(item[0])
            if item[1] not in pro:
                pro.append(item[1])
        return np.array(inters), np.array(rna), np.array(pro)

class split_train_test:
    def __init__(self, interactions_file):
        self.inters = interactions_file
    def get_labels(self):
        with open(self.inters) as pid:
            data = pid.readlines()
        inters_id = get_inters_and_id(self.inters)
        _, rna_id, pro_id = inters_id.get_iters_rna_id_pro_id()
        labels = []
        for i in rna_id:
            for j in pro_id:
                if (i + '\t' + j + '\n') in data:
                    labels.append(1)
                else:
                    labels.append(0)
        return np.array(labels)

    def split_train_test_graph(self, testing_ratio=0.2):
        inters_id = get_inters_and_id(self.inters)
        inters, rna_id, pro_id = inters_id.get_iters_rna_id_pro_id()
        nodes = np.array(list(rna_id) + list(pro_id))
        G = nx.Graph()
        G.add_nodes_from(rna_id)
        G.add_nodes_from(pro_id)
        G.add_edges_from(inters)
        node_num1, edge_num1 = len(G.nodes), len(G.edges)
        print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
        testing_edges_num = int(len(G.edges) * testing_ratio)
        tem_testing_pos_edges = random.sample(G.edges, testing_edges_num)
        testing_pos_edges = []
        G_train = copy.deepcopy(G)
        for edge in tem_testing_pos_edges:
            node_u, node_v = edge
            if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
                G_train.remove_edge(node_u, node_v)
                testing_pos_edges.append(edge)
        G_train.remove_nodes_from(nx.isolates(G_train))
        node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
        assert node_num1 == node_num2
        print('Training Graph: nodes:', node_num2, 'edges:', edge_num2)

        return G_train, testing_pos_edges, edge_num2, len(testing_pos_edges), nodes

    def load(self, i):
        datadir = os.path.join('data/1/data/dataset', str(i + 1))
        if not os.path.exists(datadir):
            G_train, test_pos_edges, train_num, test_num, nodes = self.split_train_test_graph()
            labels = self.get_labels()
            os.makedirs(datadir)
            train_num = np.array(train_num)
            test_num = np.array(test_num)
            nx.write_edgelist(G_train, f'{datadir}/train_graph_edges', data=False)
            np.save(f'{datadir}/test_pos_edges.npy', test_pos_edges)
            np.save(f'{datadir}/train_num.npy', train_num)
            np.save(f'{datadir}/test_num.npy', test_num)
            np.save(f'{datadir}/labels.npy', labels)

        else:
            test_pos_edges = np.load(f'{datadir}/test_pos_edges.npy')
            train_num = np.load(f'{datadir}/train_num.npy')
            test_num = np.load(f'{datadir}/test_num.npy')
            labels = np.load(f'{datadir}/labels.npy')
        return test_pos_edges, train_num, test_num, labels


