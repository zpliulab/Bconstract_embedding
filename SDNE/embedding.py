import networkx as nx
import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds
import torch
from model import MNN
from dataset import Dataload
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import get_inters_and_id

class node_embedding:
    def __init__(self, G_file, nodes):
        self.G_file = G_file
        self.nodes = list(nodes)

    def get_node_embeds(self):
        G_train = nx.read_adjlist(self.G_file)
        adjacency_matrix = nx.adjacency_matrix(G_train, self.nodes).todense()
        Adj = torch.FloatTensor(adjacency_matrix)
        Nodes_num = len(self.nodes)
        Model = MNN(Nodes_num, 500, 128, 0.5, 0.01)
        opt = optim.Adam(Model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)  # 每十个周期调整学习率为0.9倍
        Data = Dataload(Adj, Nodes_num)
        Data = DataLoader(Data, batch_size=100, shuffle=True, )  # 每批100
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Model = Model.to(device)
        Model.train()
        for epoch in range(200):
            for index in Data:
                adj_batch = Adj[index]  # 100*2708
                adj_mat = adj_batch[:, index]  # 100*100
                b_mat = torch.ones_like(adj_batch)  # 100*2708全为1
                b_mat[adj_batch != 0] = 5  # 5

                opt.zero_grad()
                L_1st, L_2nd, L_all = Model(adj_batch, adj_mat, b_mat)
                L_reg = 0
                for param in Model.parameters():
                    L_reg += 1e-5 * torch.sum(torch.abs(param)) + 1e-4 * torch.sum(param * param)  # torch.abs求绝对值
                Loss = L_all + L_reg
                Loss.backward()
                opt.step()
            # scheduler.step(epoch)
            # print("loss for epoch %d is:" % epoch)
            # print("loss is %f" % Loss)
        Model.eval()
        embedding = Model.savector(Adj)
        outVec = embedding.detach().numpy()
        rna_embeds = outVec[0:935]
        pro_embeds = outVec[935:]
        return rna_embeds, pro_embeds

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