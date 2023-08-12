import torch
import matplotlib.pyplot as plt

from torch_geometric.utils import dropout_adj

from .evaluate import *

import sys
sys.path.append('../../')
from pGRACE.functional import degree_drop_weights, drop_edge_weighted, drop_edge_weighted_combine
from priv_weights import sp_weights, random_walk_weights, combine_weights


class PAGCL:
    def __init__(self, data, edge_index_p, edge_label_p, edge_index_t, edge_label_t, edge_index_p_pos,
                 args, logger, t, model, optimizer, weights_matrix, use_priv_enhance=False):
        self.device = args.device
        self.dataset = args.dataset
        self.method = args.method
        self.use_attr = args.use_attr
        self.data = data
        self.num_nodes = data.y.size()[0]
        self.edge_index_t = edge_index_t
        self.edge_index_p = edge_index_p
        self.edge_label_t = edge_label_t
        self.edge_label_p = edge_label_p
        self.edge_index_p_pos = edge_index_p_pos
        self.model = model
        self.optimizer = optimizer
        self.weights_matrix = weights_matrix
        self.use_priv_enhance = use_priv_enhance
        if args.method == 'pagcl_sp':
            self.edge_weights = sp_weights(data, edge_index_p_pos, initial_weight=0.2, sp_weight=0.8)
        elif args.method == 'pagcl_rw':
            self.edge_weights = random_walk_weights(data, edge_index_p_pos)
            plt.hist(x=self.edge_weights, bins=20, color="steelblue", edgecolor="black")
            plt.show()
        elif args.method == 'combine':
            de_weights = degree_drop_weights(data.edge_index)
            self.de_weights = de_weights
            self.edge_weights = combine_weights(de_weights, data, edge_index_p_pos, k=1)
        else:
            self.edge_weights = None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        # how to drop edge?
        if self.method == 'random':
            edge_index_1 = dropout_adj(self.data.edge_index, p=0.3)[0]
            edge_index_2 = dropout_adj(self.data.edge_index, p=0.4)[0]
        elif self.method == 'gca':
            drop_weights = degree_drop_weights(self.data.edge_index)
            edge_index_1 = drop_edge_weighted(self.data.edge_index, drop_weights, p=0.3, threshold=0.7)
            edge_index_2 = drop_edge_weighted(self.data.edge_index, drop_weights, p=0.4, threshold=0.7)
        elif self.method == 'pagcl_sp':
            sp_weights = self.edge_weights
            masks1 = torch.bernoulli(1 - torch.Tensor(sp_weights)).to(torch.bool)
            masks2 = torch.bernoulli(1 - torch.Tensor(sp_weights)).to(torch.bool)
            edge_index_1 = self.data.edge_index[:, masks1]
            edge_index_2 = self.data.edge_index[:, masks2]
            print(edge_index_1.size())
            print(edge_index_2.size())
        elif self.method == 'pagcl_rw':
            rw_weights = self.edge_weights
            masks1 = torch.bernoulli(1 - torch.Tensor(rw_weights)).to(torch.bool)
            masks2 = torch.bernoulli(1 - torch.Tensor(rw_weights)).to(torch.bool)
            print(masks1.sum())
            print(masks2.sum())
            edge_index_1 = self.data.edge_index[:, masks1]
            edge_index_2 = self.data.edge_index[:, masks2]
            # rw_weights = torch.Tensor(rw_weights)
            # edge_index_1 = drop_edge_weighted(self.data.edge_index, rw_weights, p=0.3, threshold=0.7)
            # edge_index_2 = drop_edge_weighted(self.data.edge_index, rw_weights, p=0.4, threshold=0.7)
            # print(edge_index_1.size())
            # print(edge_index_2.size())

        elif self.method == 'combine':
            edge_index_1 = drop_edge_weighted_combine(self.edge_weights, self.data.edge_index, self.de_weights, p=0.3,
                                               threshold=0.7)
            edge_index_2 = drop_edge_weighted_combine(self.edge_weights, self.data.edge_index, self.de_weights, p=0.3,
                                                      threshold=0.7)
        else:
            print("Not Defined Method. Please confirm the method.")
            edge_index_1 = dropout_adj(self.data.edge_index, p=0.3)[0]
            edge_index_2 = dropout_adj(self.data.edge_index, p=0.4)[0]

        if self.data.x is None or self.use_attr is False:
            # print("Feature is None or not use feature.")
            x_1 = torch.eye(self.num_nodes)
            x_2 = torch.eye(self.num_nodes)
        else:
            x_1 = self.data.x
            x_2 = self.data.x

        z1 = self.model(x_1, edge_index_1)
        z2 = self.model(x_2, edge_index_2)

        if self.use_priv_enhance:
            loss = self.model.loss_priv(z1, z2, self.weights_matrix, batch_size=1024 if self.dataset == 'Coauthor-Phy' else None)
        else:
            loss = self.model.loss(z1, z2, batch_size=1024 if self.dataset == 'Coauthor-Phy' else None)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test(self, data):
        with torch.no_grad():
            self.model.eval()
            if data.x is None or self.use_attr is False:
                embed = self.model(torch.eye(self.num_nodes), data.edge_index)
            else:
                embed = self.model(data.x, data.edge_index)
            link_auc = link_pred_cos(embed, self.edge_index_t, self.edge_label_t)
            attack_auc = link_pred_cos(embed, self.edge_index_p, self.edge_label_p)
            node_acc = node_cls(embed, data.y)

        return link_auc, node_acc, attack_auc

    # def attacks(self, data):
    #     with torch.no_grad():
    #         self.model.eval()
    #         embed = self.model(data.x, data.edge_index)
    #         embed = embed.detach().cpu().numpy()
    #         sim_attacks(embed,self.)

