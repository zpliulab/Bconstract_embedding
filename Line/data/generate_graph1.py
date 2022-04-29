import numpy as np
import copy
import networkx as nx
import random
import os


def get_iters_rna_id_pro_id(interactions_file):
    with open(interactions_file) as pid:
        data = pid.readlines()
    rna = []
    pro = []
    inters = []
    for item in data:
        eve_inter = []
        item = item.strip().split('\t')
        eve_inter.append(item[0])
        eve_inter.append(item[1])
        eve_inter.append({'weight':1.0})
        inters.append(eve_inter)

        if item[0] not in rna:
            rna.append(item[0])
        if item[1] not in pro:
            pro.append(item[1])
    return np.array(inters), np.array(rna), np.array(pro)


def split_train_test_graph(fold, inter_file_path, fold_path = None, testing_ratio=0.2):
    inters, rna_id, pro_id = get_iters_rna_id_pro_id(inter_file_path)
    nodes = np.array(list(rna_id) + list(pro_id))
    G = nx.Graph()
    G.add_nodes_from(rna_id)
    G.add_nodes_from(pro_id)
    G.add_edges_from(inters)
    node_num1, edge_num1 = len(G.nodes()), len(G.edges())
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges()) * testing_ratio)
    random.seed(fold)
    tem_testing_pos_edges = random.sample(G.edges(), testing_edges_num)
    testing_pos_edges = []
    G_train = copy.deepcopy(G)
    for edge in tem_testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)
            testing_pos_edges.append(edge)
    G_train.remove_nodes_from(nx.isolates(G_train))
    nx.write_gpickle(G_train, f'{fold_path}/G_train')

if __name__ == '__main__':
    for fold in range(3):
        fold_path = '5/fold' + str(fold + 1)
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
        split_train_test_graph(fold, '5/inters.txt', fold_path= fold_path)