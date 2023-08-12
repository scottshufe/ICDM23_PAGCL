import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import LastFMAsia, FacebookPagePage, Planetoid, PolBlogs
from torch_geometric.utils import remove_self_loops, negative_sampling
from torch_geometric.data import Data
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.metrics import roc_auc_score, average_precision_score


def hadamard(x, y):
    return x*y


def l1_weight(x, y):
    return np.absolute(x-y)


def l2_weight(x, y):
    return np.square(x-y)


def concate(x, y):
    return np.concatenate((x, y), axis=1)


def average(x, y):
    return (x+y)/2


def get_dataset(name, transform=None, root='data/'):
    if name == 'LastFMAsia':
        dataset = LastFMAsia(root + name, transform=T.NormalizeFeatures())
    elif name == 'FacebookPagePage':
        dataset = FacebookPagePage(root + name, transform=T.NormalizeFeatures())
    elif name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root, name, transform=transform if transform is not None else T.NormalizeFeatures())
    elif name == 'PolBlogs':
        dataset = PolBlogs(root + name)
    else:
        raise NotImplementedError
    dataset.data.edge_index, _ = remove_self_loops(dataset.data.edge_index)
    return dataset


def preprocess(o_data, logger, p_ratio=0.1, t_ratio=0.1, node_train_ratio=0.3, seed=42):
    if o_data.x != None:
        num_n = o_data.x.shape[0]
    else:
        num_n = o_data.num_nodes
    train_mask = torch.zeros(num_n, dtype=torch.bool)
    test_mask = torch.zeros(num_n, dtype=torch.bool)
    num_train = int(num_n * node_train_ratio)

    torch.manual_seed(seed)
    perm = torch.randperm(num_n)
    train_mask[perm[:num_train]] = True
    test_mask[perm[num_train:]] = True

    edge_index = o_data.edge_index[:, o_data.edge_index[0] <= o_data.edge_index[1]]
    num_e = edge_index.shape[1]

    torch.manual_seed(seed)
    idx = torch.randperm(num_e)
    num_p, num_t = int(p_ratio * num_e), int(t_ratio * num_e)
    idx1 = idx[: num_e - num_p - num_t]
    idx_p = idx[num_e - num_p - num_t: num_e - num_t]
    idx_t = idx[num_e - num_t:]
    edge_index1 = edge_index[:, idx1]
    edge_index_p_pos = edge_index[:, idx_p]
    edge_index_t_pos = edge_index[:, idx_t]

    edge_index_p_neg = negative_sampling(edge_index=o_data.edge_index, num_nodes=o_data.num_nodes,
                                         num_neg_samples=edge_index_p_pos.size(1), method='sparse')
    edge_index_p = torch.cat([edge_index_p_pos, edge_index_p_neg], dim=-1)
    edge_label_p = torch.cat([torch.ones(edge_index_p_pos.size(1)), torch.zeros(edge_index_p_neg.size(1))], dim=0)

    edge_index_t_neg = negative_sampling(edge_index=o_data.edge_index, num_nodes=o_data.num_nodes,
                                         num_neg_samples=edge_index_t_pos.size(1), method='sparse')
    edge_index_t = torch.cat([edge_index_t_pos, edge_index_t_neg], dim=-1)
    edge_label_t = torch.cat([torch.ones(edge_index_t_pos.size(1)), torch.zeros(edge_index_t_neg.size(1))], dim=0)

    data = Data(x=o_data.x, y=o_data.y, train_mask=train_mask, test_mask=test_mask,
                edge_index_half=edge_index1, edge_index=torch.cat((edge_index1, edge_index1[[1, 0], :]), 1))

    logger.info(f'train nodes: {train_mask.sum().item()}, test nodes: {test_mask.sum().item()}, '
                f'public edges: {edge_index1.shape[1]} * 2, private edges: {edge_index_p_pos.shape[1]} * 2, '
                f'private nodes: {len(set(edge_index_p_pos.cpu().detach().numpy()[0]))}')
    return data, edge_index_p_pos, edge_index_p, edge_label_p, edge_index_t_pos, edge_index_t, edge_label_t


def sim_attacks(embs, test_edges):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list = [[] for _ in range(len(sim_metric_list))]
    for edge in test_edges:
        for j in range(len(sim_metric_list)):
            sim = sim_metric_list[j](embs[edge[0]], embs[edge[1]])
            sim_list[j].append(sim)
    return sim_list


def get_sim_auc(i, sim_list, label):
    pred = np.array(sim_list[i], dtype=np.float64)
    where_are_nan = np.isnan(pred)
    where_are_inf = np.isinf(pred)
    pred[where_are_nan] = 0
    pred[where_are_inf] = 0

    i_auc = roc_auc_score(label, pred)
    print(i_auc)
    i_ap = average_precision_score(label, pred)
    if i_auc < 0.5:
        i_auc = 1 - i_auc
    if i_ap < 0.5:
        i_ap = 1 - i_ap

    return i_auc


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

