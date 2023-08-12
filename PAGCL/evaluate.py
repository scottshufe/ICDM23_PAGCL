import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import negative_sampling
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


def link_pred_cos(embed, edge_index, edge_label):
    embed_norm = F.normalize(embed, dim=1)
    pred = (embed_norm[edge_index[0]] * embed_norm[edge_index[1]]).sum(dim=-1).view(-1).cpu().detach().numpy()
    label = edge_label.cpu().detach().numpy()
    return roc_auc_score(label, pred) * 100


def link_pred_ml(embed, data, edge_index_test, edge_label_test):
    embed_norm = F.normalize(embed, dim=1)
    neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                       num_neg_samples=data.edge_index.size(1), method='sparse')
    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    train_x = torch.cat([embed_norm[edge_label_index[0]], embed_norm[edge_label_index[1]]], dim=1).cpu().detach().numpy()
    train_y = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).cpu().detach().numpy()
    test_x = torch.cat([embed_norm[edge_index_test[0]], embed_norm[edge_index_test[1]]], dim=1).cpu().detach().numpy()
    test_y = edge_label_test.cpu().detach().numpy()
    classifier = SVC(C=10)
    classifier.fit(train_x, train_y)
    return roc_auc_score(test_y, classifier.predict(test_x)) * 100


def node_cls(embed, y):
    cls = LogisticRegression()
    cv_results = cross_validate(cls, embed, y, scoring=['accuracy'], cv=5)
    accs = cv_results['test_accuracy']
    acc = np.mean(accs)
    return acc

def sim_attacks(embs, test_edges):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list = [[] for _ in range(len(sim_metric_list))]
    for edge in test_edges:
        for j in range(len(sim_metric_list)):
            sim = sim_metric_list[j](embs[edge[0]], embs[edge[1]])
            sim_list[j].append(sim)
    return sim_list


def write_auc(dataset, epoch, write_path, sim_list, label):
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']

    auc_list = []
    for i in range(len(sim_list_str)):
        with open(write_path + "/sim_attacks_{}.txt".format(sim_list_str[i]), "a") as wf:
            pred = np.array(sim_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            i_auc = roc_auc_score(label, pred)
            i_ap = average_precision_score(label, pred)
            if i_auc < 0.5:
                i_auc = 1 - i_auc
            if i_ap < 0.5:
                i_ap = 1 - i_ap
            print(sim_list_str[i], i_auc, i_ap)
            wf.write(
                "%s,%s,%d,%0.5f,%0.5f\n" %
                (dataset, "%s" %
                 (sim_list_str[i]), epoch, i_auc, i_ap))
            auc_list.append(i_auc)
    max_auc = max(auc_list)
    index_max = np.argmax(auc_list)
    return sim_list_str[index_max], max_auc
