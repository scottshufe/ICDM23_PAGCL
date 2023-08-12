import torch
import random
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from collections import Counter
from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, to_undirected, to_networkx, remove_self_loops, to_dense_adj
import time
from tqdm import tqdm
import logging
from scipy import sparse as sp


def sp_weights(data, private_edge_index, initial_weight=0.3, sp_weight=0.9):
    data.num_nodes = data.y.size()[0]
    edge_index = data.edge_index.cpu().numpy()
    G = to_networkx(data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(private_edge_index.size()[1]):
        n1 = private_edge_index[:, i:i + 1].flatten()[0].item()
        n2 = private_edge_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # all links, and all shortest paths, -> find all links on shortest paths
    sp_edge_list = []
    sp_edge_dict = {}
    for e in priv_edge_list:
        try:
            sp = nx.shortest_path(G, e[0], e[1])
            for i in range(len(sp) - 1):
                sp_edge_list.append([sp[i], sp[i+1]])
                if (sp[i], sp[i+1]) not in sp_edge_dict:
                    sp_edge_dict[(sp[i], sp[i+1])] = 1
                else:
                    sp_edge_dict[(sp[i], sp[i + 1])] += 1
        except:
            # no path exist
            continue

    count_dict = {}
    num_link = edge_index.shape[1]
    for i in range(num_link):
        edge = (edge_index[0, i], edge_index[1, i])
        if edge in sp_edge_dict:
            count_dict[edge] = sp_edge_dict[edge]
        else:
            count_dict[edge] = 0

    count_res = Counter(list(count_dict.values()))
    print(count_res)
    # xs = []
    # ys = []
    # for x in sorted(count_res.keys()):
    #     y = count_res[x]
    #     xs.append(x)
    #     ys.append(y)
    # plt.bar(x=xs, height=ys)
    # plt.show()

    # print(xxx)

    weights = []
    for i in range(num_link):
        edge = (edge_index[0, i], edge_index[1, i])
        if count_dict[edge] == 0:
            weights.append(initial_weight)
        else:
            weights.append(sp_weight)


    # df1 = pd.DataFrame(np.array(sp_edge_list))
    # df2 = pd.DataFrame({0: df1[1], 1: df1[0]})
    # df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
    #
    # sp_links = []
    # for i in range(len(df)):
    #     sp_links.append(str(df[0][i]) + '-' + str(df[1][i]))
    #
    # masks = []
    # weights = []
    # for i in range(data.edge_index.shape[1]):
    #     n1 = data.edge_index[0][i].item()
    #     n2 = data.edge_index[1][i].item()
    #     link = str(n1) + '-' + str(n2)
    #     if link in sp_links:
    #         masks.append(0)
    #         weights.append(sp_weight)
    #     else:
    #         masks.append(1)
    #         weights.append(initial_weight)
    # masks = torch.Tensor(masks).to(torch.bool)

    return weights


def shortest_path_weights(train_data, test_data, strategy='simple', rm_edge_num=1, initial_weight=0.3, sp_weight=0.9):
    G = to_networkx(train_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(int(len(test_data.edge_label) / 2)):
        n1 = test_data.edge_label_index[:, i:i + 1].flatten()[0].item()
        n2 = test_data.edge_label_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 所有隐私边，所有最短路径
    sp_edge_list = []
    sp_lens = []
    for e in priv_edge_list:
        try:
            # for sp in nx.all_shortest_paths(G, e[0], e[1]):
            sp = nx.shortest_path(G, e[0], e[1])
            # print(e[0], e[1], len(sp), sp)
            sp_lens.append(len(sp))
            if strategy == 'simple' or strategy == 'betweenness':
                # 保存所有最短路径上的边
                for i in range(len(sp) - 1):
                    sp_edge_list.append([sp[i], sp[i + 1]])
            elif strategy == 'threshold':
                # print('run here')
                # 如果最短路径长度小于等于设定的边数量，全部保存
                if (len(sp) - 1) <= rm_edge_num:
                    for i in range(len(sp) - 1):
                        sp_edge_list.append([sp[i], sp[i + 1]])
                # 否则随机抽取设定数量的边保存
                else:
                    arr = np.random.choice(len(sp) - 1, size=rm_edge_num, replace=False)
                    for i in arr:
                        sp_edge_list.append([sp[i], sp[i + 1]])
            else:
                raise NotImplementedError('strategy = {} not implemented!'.format(strategy))
        except:
            # print("no path exist.")
            pass

    # print('len sp_edge_list ', len(sp_edge_list))
    print('average shortest path length: {}'.format(np.mean(sp_lens)))

    df1 = pd.DataFrame(np.array(sp_edge_list))
    df2 = pd.DataFrame({0: df1[1], 1: df1[0]})

    if strategy == 'simple' or 'threshold':
        df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        sp_links = []
        for i in range(len(df)):
            sp_links.append(str(df[0][i]) + '-' + str(df[1][i]))
        print('len sp links ', len(sp_links))
        weights = []
        for i in range(train_data.edge_index.shape[1]):
            n1 = train_data.edge_index[0][i].item()
            n2 = train_data.edge_index[1][i].item()
            link = str(n1) + '-' + str(n2)
            if link in sp_links:
                weights.append(sp_weight)
            else:
                weights.append(initial_weight)

    elif strategy == 'betweenness':
        df = pd.concat([df1, df2]).reset_index(drop=True)
        df = df.rename(columns={0: 'n1', 1: 'n2'})
        df_count = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
        btn_weights = betweenness_weight_func(df_count['count'])
        sp_links_weight_dict = {}
        for i in range(len(df_count)):
            key = str(df_count['n1'][i]) + '-' + str(df_count['n2'][i])
            sp_links_weight_dict[key] = btn_weights[i]
        weights = []
        for i in range(train_data.edge_index.shape[1]):
            n1 = train_data.edge_index[0][i].item()
            n2 = train_data.edge_index[1][i].item()
            link = str(n1) + '-' + str(n2)
            if link in sp_links_weight_dict:
                weights.append(sp_links_weight_dict[link])
            else:
                weights.append(initial_weight)
    else:
        raise NotImplementedError('strategy = {} not implemented!'.format(strategy))

    return weights


def shortest_path_weights_2(train_data, test_data, strategy='simple', rm_edge_num=1, initial_weight=0.3, sp_weight=0.9):
    G = to_networkx(train_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(int(len(test_data.edge_label) / 2)):
        n1 = test_data.edge_label_index[:, i:i + 1].flatten()[0].item()
        n2 = test_data.edge_label_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 所有隐私边，所有最短路径
    sp_edge_list = []
    sp_lens = []
    for e in priv_edge_list:
        try:
            sp = nx.shortest_path(G, e[0], e[1])
            sp_lens.append(len(sp))
            for sp in nx.all_shortest_paths(G, e[0], e[1]):
            # sp = nx.shortest_path(G, e[0], e[1])
            # print(e[0], e[1], len(sp), sp)
            # sp_lens.append(len(sp))
                if strategy == 'simple':
                    # 保存所有最短路径上的边
                    for i in range(len(sp) - 1):
                        sp_edge_list.append([sp[i], sp[i + 1]])
                else:
                    raise NotImplementedError('strategy = {} not implemented!'.format(strategy))
        except:
            # print("no path exist.")
            pass

    # print('len sp_edge_list ', len(sp_edge_list))
    print('average shortest path length: {}'.format(np.mean(sp_lens)))

    df1 = pd.DataFrame(np.array(sp_edge_list))
    df2 = pd.DataFrame({0: df1[1], 1: df1[0]})

    if strategy == 'simple':
        df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        sp_links = []
        for i in range(len(df)):
            sp_links.append(str(df[0][i]) + '-' + str(df[1][i]))
        print('len sp links ', len(sp_links))
        weights = []
        for i in range(train_data.edge_index.shape[1]):
            n1 = train_data.edge_index[0][i].item()
            n2 = train_data.edge_index[1][i].item()
            link = str(n1) + '-' + str(n2)
            if link in sp_links:
                weights.append(sp_weight)
            else:
                weights.append(initial_weight)
    else:
        raise NotImplementedError('strategy = {} not implemented!'.format(strategy))

    return weights


def shortest_path_weights_new(train_data, test_data, initial_weight=0.3, sp_weight=0.9):
    G = to_networkx(train_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(int(len(test_data.edge_label) / 2)):
        n1 = test_data.edge_label_index[:, i:i + 1].flatten()[0].item()
        n2 = test_data.edge_label_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 所有隐私边，所有最短路径
    sp_edge_list = []
    sp_lens = []
    for e in priv_edge_list:
        try:
            sp = nx.shortest_path(G, e[0], e[1])
            sp_lens.append(len(sp))
            for sp in nx.all_shortest_paths(G, e[0], e[1]):
            # sp = nx.shortest_path(G, e[0], e[1])
            # print(e[0], e[1], len(sp), sp)
            # sp_lens.append(len(sp))
                for i in range(len(sp) - 1):
                    if sp[i] < sp[i+1]:
                        sp_edge_list.append(str(sp[i]) + '-' + str(sp[i + 1]))
                    else:
                        sp_edge_list.append(str(sp[i+1]) + '-' + str(sp[i]))
                    # sp_edge_list.append([sp[i], sp[i + 1]])
        except:
            # print("no path exist.")
            pass

    # print('len sp_edge_list ', len(sp_edge_list))
    print('average shortest path length: {}'.format(np.mean(sp_lens)))

    sp_edge_dict = {}
    for i in range(len(sp_edge_list)):
        sp_link = sp_edge_list[i]
        if sp_link in sp_edge_dict:
            sp_edge_dict[sp_link] += 1
        else:
            sp_edge_dict[sp_link] = 1


    for i in range(len(sp_edge_list)):
        ns = sp_edge_list[i].split('-')
        sp_link = ns[1] + '-' + ns[0]
        sp_edge_dict[sp_link] = sp_edge_dict[sp_edge_list[i]]

    sp_col = []
    for i in range(train_data.edge_index.shape[1]):
        n1 = train_data.edge_index[0][i].item()
        n2 = train_data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        if link in sp_edge_dict:
            sp_col.append(sp_edge_dict[link])
        else:
            sp_col.append(0)

    sp_col = torch.tensor(sp_col).to(torch.float32)

    sp_col = torch.log(sp_col + 1)
    weights = sp_col / (sp_col.max() - sp_col.mean())

    return weights


def betweenness_weight_func(counts):
    weights = []
    max_count = max(counts)
    for count in counts:
        weight = 0.5 + count / (2 * max_count)
        weights.append(0.5 + count / 2 * max_count)
    return weights


def emb_perturb(embeddings, mu=0, b=0.1):
    size = embeddings.shape
    noise = np.random.laplace(mu, b, size)
    embeddings = embeddings + noise

    return embeddings


def common_neighbor_weights(G, train_data, test_data, initial_weight=0.3, cn_weight=0.9):
    priv_edge_list = []
    for i in range(int(len(test_data.edge_label) / 2)):
        n1 = test_data.edge_label_index[:, i:i + 1].flatten()[0].item()
        n2 = test_data.edge_label_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 先从原graph中移除所有隐私边
    G.remove_edges_from(priv_edge_list)

    # 所有隐私边，所有共同好友 common_friends
    cn_nodes_dict = {}
    cn_edge_list = []

    n = 0
    for e in priv_edge_list:
        try:
            cn = list(nx.common_neighbors(G, e[0], e[1]))
            print(e[0], e[1], cn)
            cn_nodes_dict[str(e[0]) + '+' + str(e[1])] = cn
            if len(cn) == 0:
                n += 1
            else:
                for node in cn:
                    cn_edge_list.append([e[0], node])
                    cn_edge_list.append([node, e[1]])
        except:
            print("error.")

    df1 = pd.DataFrame(np.array(cn_edge_list))
    df2 = pd.DataFrame({
        0: df1[1],
        1: df1[0]
    })
    df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

    cn_links = []
    for i in range(len(df)):
        cn_links.append(str(df[0][i]) + '-' + str(df[1][i]))

    weights = []
    for i in range(train_data.edge_index.shape[1]):
        n1 = train_data.edge_index[0][i].item()
        n2 = train_data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        if link in cn_links:
            weights.append(cn_weight)
        else:
            weights.append(initial_weight)

    return weights


def edge_rand(adj, epsilon=10, noise_seed=42):
    s = 2 / (np.exp(epsilon) + 1)
    # print(f's = {s:.4f}')

    N = adj.shape[0]
    t = time.time()

    # np.random.seed(noise_seed)
    bernoulli = np.random.binomial(1, s, (N, N))

    # print(f'generating perturbing vector done using {time.time() - t} secs!')
    # logging.info(f'generating perturbing vector done using {time.time() - t} secs!')

    # find the randomization entries
    entry = np.asarray(list(zip(*np.where(bernoulli))))

    # print("the number of the flipping entries: ", len(entry))

    dig_1 = np.random.binomial(1, 1 / 2, len(entry))
    indice_1 = entry[np.where(dig_1 == 1)[0]]
    indice_0 = entry[np.where(dig_1 == 0)[0]]

    add_mat = construct_sparse_mat(indice_1, N)
    minus_mat = construct_sparse_mat(indice_0, N)

    adj_noisy = adj + add_mat - minus_mat

    adj_noisy.data[np.where(adj_noisy.data == -1)] = 0
    adj_noisy.data[np.where(adj_noisy.data == 2)] = 1

    return adj_noisy


def construct_sparse_mat(indice, N):
    cur_row = -1
    new_indices = []
    new_indptr = []

    # for i, j in tqdm(indice):
    for i, j in indice:
        if i >= j:
            continue

        while i > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        new_indices.append(j)

    while N > cur_row:
        new_indptr.append(len(new_indices))
        cur_row += 1

    data = np.ones(len(new_indices), dtype=np.int64)
    indices = np.asarray(new_indices, dtype=np.int64)
    indptr = np.asarray(new_indptr, dtype=np.int64)

    mat = sp.csr_matrix((data, indices, indptr), (N, N))

    return mat + mat.T


def lap_graph(adj, epsilon=10, noise_type='laplace', noise_seed=42, delta=1e-5):
    n_nodes = adj.shape[0]
    n_edges = adj.sum() / 2

    N = n_nodes
    t = time.time()

    A = sp.tril(adj, k=-1)
    # print('getting the lower triangle of adj matrix done!')

    eps_1 = epsilon * 0.01
    eps_2 = epsilon - eps_1
    noise = get_noise(noise_type=noise_type, size=(N, N), seed=noise_seed,
                      eps=eps_2, delta=delta, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=bool)
    # print(f'generating noise done using {time.time() - t} secs!')

    A += noise
    # print(f'adding noise to the adj matrix done!')

    t = time.time()
    while True:
        n_edges_keep = int(n_edges + int(
            get_noise(noise_type=noise_type, size=1, seed=noise_seed,
                      eps=eps_1, delta=delta, sensitivity=1)[0]))
        if n_edges_keep > 0:
            break
        else:
            continue
    # print(f'edge number from {n_edges} to {n_edges_keep}')

    t = time.time()
    a_r = A.A.ravel()

    n_splits = 50
    len_h = len(a_r) // n_splits
    ind_list = []
    # for i in tqdm(range(n_splits - 1)):
    for i in range(n_splits - 1):
        ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i)

    ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert (col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    # print(f'data preparation done using {time.time() - t} secs!')

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    return mat + mat.T


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    # np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity / eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2 * np.log(1.25 / delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


def combine_weights(edge_weights, data, private_edge_index, k=1):
    """
    edge_weights: 根据中心性计算得到的删除权重，越高的越不重要，删除概率越高
    k: 每条最短路径删除 topk 的边
    """

    # 用一个字典来存储每一条边的权重: 这里权重越高，在图中越不重要，越应该被扔掉
    weight_dict = {}
    for i in range(data.edge_index.shape[1]):
        n1 = data.edge_index[0][i].item()
        n2 = data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        weight_dict[link] = edge_weights[i]

    # 构建Graph
    num_node = data.x.size()[0]
    tmp_data = Data(edge_index=data.edge_index, num_nodes=num_node)
    G = to_networkx(tmp_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(int(private_edge_index.size()[1] / 2)):
        n1 = private_edge_index[:, i:i + 1].flatten()[0].item()
        n2 = private_edge_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 找到每条最短路径上被移除权重最大的边
    drop_sp_link_list = []
    for e in priv_edge_list:
        try:
            sp = nx.shortest_path(G, e[0], e[1])
            # print(e[0], e[1], sp)

            sp_link_weights = []
            sp_links = []

            for i in range(len(sp) - 1):
                link = str(sp[i]) + '-' + str(sp[i + 1])
                sp_links.append([sp[i], sp[i + 1]])

                link_weight = weight_dict[link]
                sp_link_weights.append(link_weight)
            sp_link_weights = torch.Tensor([sp_link_weights])

            # top k links to be deleted
            if len(sp) <= k + 1:
                for i in range(len(sp)):
                    drop_sp_link_list.append(sp_links[i])
            else:
                topk_link_weight_idx = torch.topk(sp_link_weights, k).indices
                for idx in topk_link_weight_idx.flatten():
                    drop_sp_link_list.append(sp_links[i])
        except:
            print("no path exist.")

    df1 = pd.DataFrame(np.array(drop_sp_link_list))
    df2 = pd.DataFrame({0: df1[1], 1: df1[0]})
    df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

    drop_sp_link_list = []
    for i in range(len(df)):
        drop_sp_link_list.append(str(df[0][i]) + '-' + str(df[1][i]))
    print('len sp links ', len(drop_sp_link_list))

    sp_masks = []
    for i in range(data.edge_index.shape[1]):
        n1 = data.edge_index[0][i].item()
        n2 = data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        # drop link in drop list
        if link in drop_sp_link_list:
            sp_masks.append(True)
        else:
            # keep link not in drop list
            sp_masks.append(False)

    sp_masks = torch.Tensor(sp_masks).bool()

    return sp_masks


def combine_weights_v2(edge_weights, train_data, test_data, k=1):
    """
    edge_weights: 根据中心性计算得到的删除权重，越高的越不重要，删除概率越高
    k: 每条最短路径删除 topk 的边
    """

    # 用一个字典来存储每一条边的权重: 这里权重越高，在图中越不重要，越应该被扔掉
    weight_dict = {}
    for i in range(train_data.edge_index.shape[1]):
        n1 = train_data.edge_index[0][i].item()
        n2 = train_data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        weight_dict[link] = edge_weights[i]

    # 构建 Graph
    G = to_networkx(train_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    priv_edge_list = []
    for i in range(int(len(test_data.edge_label) / 2)):
        n1 = test_data.edge_label_index[:, i:i + 1].flatten()[0].item()
        n2 = test_data.edge_label_index[:, i:i + 1].flatten()[1].item()
        priv_edge_list.append([n1, n2])

    # 迭代地找到每条最短路径上被移除权重最大的边
    drop_sp_link_list = []
    no_sp_path_1 = 0
    no_sp_path_2 = 0
    no_sp_path_3 = 0
    for e in priv_edge_list:
        tmp_G = G.copy()
        for i in range(k):
            try:
                sp = nx.shortest_path(tmp_G, e[0], e[1])
            except:
                print("iter {}, no path exist.".format(i + 1))
                if i + 1 == 1:
                    no_sp_path_1 += 1
                elif i + 1 == 2:
                    no_sp_path_2 += 1
                elif i + 1 == 3:
                    no_sp_path_3 += 1
                break
            sp_links = []
            sp_link_weights = []
            for j in range(len(sp) - 1):
                link = str(sp[j]) + '-' + str(sp[j + 1])
                sp_links.append([sp[j], sp[j + 1]])
                link_weight = weight_dict[link]
                sp_link_weights.append(link_weight)
            sp_link_weights = torch.Tensor([sp_link_weights])

            # find the top 1 link
            max_idx = torch.argmax(sp_link_weights)
            drop_link = sp_links[max_idx]
            drop_sp_link_list.append(sp_links[max_idx])

            # drop edge from tmp_G
            tmp_G.remove_edge(drop_link[0], drop_link[1])

    df1 = pd.DataFrame(np.array(drop_sp_link_list))
    df2 = pd.DataFrame({0: df1[1], 1: df1[0]})
    df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

    drop_sp_link_list = []
    for i in range(len(df)):
        drop_sp_link_list.append(str(df[0][i]) + '-' + str(df[1][i]))
    print('len sp links ', len(drop_sp_link_list))
    print('no sp path iter 1: ', no_sp_path_1)
    print('no sp path iter 2: ', no_sp_path_2)
    print('no sp path iter 3: ', no_sp_path_3)

    sp_masks = []
    for i in range(train_data.edge_index.shape[1]):
        n1 = train_data.edge_index[0][i].item()
        n2 = train_data.edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        # drop link in drop list
        if link in drop_sp_link_list:
            sp_masks.append(True)
        else:
            # keep link not in drop list
            sp_masks.append(False)

    sp_masks = torch.Tensor(sp_masks).bool()

    return sp_masks


def dice_aug(train_data, test_data, p):
    edge_index = train_data.edge_index

    # find all private nodes
    true_edges = test_data.edge_label_index[:, test_data.edge_label.bool()].cpu().T.numpy()
    nodes = []
    for edge in true_edges:
        nodes.append(edge[0])
        nodes.append(edge[1])
    private_nodes = list(set(nodes))
    # find all edges connect to private nodes
    G = to_networkx(train_data)
    G = nx.to_undirected(G)
    G = nx.Graph(G)

    target_edges = np.asarray(list(G.edges(private_nodes)))

    # amount of perturbing edges
    probs = [p] * edge_index.shape[1]
    amount = int(torch.bernoulli(torch.Tensor(probs)).to(torch.bool).sum().item() / 2)

    # randomly remove or add
    adds_or_deletes = np.random.choice([0, 1], amount, p=[0.5, 0.5])
    delete_num = amount - adds_or_deletes.sum()
    add_num = adds_or_deletes.sum()

    # delete edges
    delete_idxs = np.random.choice(len(target_edges), size=delete_num, replace=False)
    delete_edges = target_edges[delete_idxs]
    delete_edges_list = []
    for e in delete_edges:
        delete_edges_list.append(str(e[0]) + "-" + str(e[1]))
        delete_edges_list.append(str(e[1]) + "-" + str(e[0]))
    delete_masks = []
    for i in range(edge_index.shape[1]):
        n1 = edge_index[0][i].item()
        n2 = edge_index[1][i].item()
        link = str(n1) + '-' + str(n2)
        # drop link in drop list
        if link in delete_edges_list:
            delete_masks.append(False)
        # keep link not in drop list
        else:
            delete_masks.append(True)

    delete_masks = torch.Tensor(delete_masks).bool()
    edge_index = edge_index[:, delete_masks]

    # add edges
    N = train_data.num_nodes
    edge_index_to_add = torch.randint(0, N, (2, add_num), dtype=torch.long)
    edge_index_to_add = to_undirected(edge_index_to_add).to(edge_index.device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

    edge_index = remove_self_loops(edge_index)[0]
    print('ori edge num: ', train_data.edge_index.shape[1])
    print('delete num: ', delete_num)
    print('add num: ', add_num)
    print('final edge num: ', edge_index.shape[1])
    return edge_index


def get_priv_adj(test_data, num_nodes):
    """
    根据输入的 private edges 产生一个标识了 private neighbors 的邻接矩阵
    param:
        private_edge_index: private edges
        num_nodes: number of nodes in original graph
    return:
        priv_adj: matrix indicates private neighbors
    """
    priv_edge_num = int(len(test_data.edge_label) / 2)
    edge_index = test_data.edge_label_index[:, :priv_edge_num]
    priv_adj = to_dense_adj(to_undirected(edge_index), max_num_nodes=num_nodes)[0]

    return priv_adj


def get_weight_matrix(priv_adj, weight=1):
    ones_matrix = torch.ones_like(priv_adj)
    w_mat = ones_matrix * weight
    weight_matrix = torch.where(priv_adj == 1, w_mat, ones_matrix)

    return weight_matrix


def random_walk_weights(data, priv_edge_index, walk_length=6, num_walks=10000, lam=1, cutoff=True):
    num_nodes = data.y.size()[0]
    # public links
    edge_index1 = data.edge_index.cpu().numpy()
    # private links
    edge_index2 = torch.cat((priv_edge_index, priv_edge_index[[1, 0], :]), 1)
    edge_index2 = edge_index2.cpu().numpy()

    num_e1 = edge_index1.shape[1]
    num_e2 = edge_index2.shape[1]
    pnodes = edge_index2[0, :]
    pnodes = list(set(pnodes))

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index1.T)

    pG = nx.Graph()
    pG.add_nodes_from(range(num_nodes))
    pG.add_edges_from(edge_index2.T)

    pdict = {}
    for i in pnodes:
        pdict[i] = list(pG.neighbors(i))

    edge_dict = {}
    for i in range(num_e1):
        edge = (edge_index1[0, i], edge_index1[1, i])
        if edge not in edge_dict.keys():
            edge_dict[edge] = 0

    # start random walking
    num = 0
    while num < num_walks:
        p = random.choice(pnodes)
        walk = [p]
        cur = p
        for j in range(walk_length):
            p_s = list(G.neighbors(cur))
            if not p_s:
                break
            cur = random.choice(p_s)
            walk.append(cur)
            if cur in pdict[p]:
                num += 1
                w_l = len(walk) - 1
                for k in range(w_l):
                    edge_dict[(walk[k], walk[k+1])] += math.exp(-w_l)
                    edge_dict[(walk[k + 1], walk[k])] += math.exp(-w_l)
                if cutoff:
                    break
    print('top10 importance score of public links:')
    for a in sorted(edge_dict.values(), reverse=True)[:10]:
        print("{:.4f}".format(a), end=' ')
    print()

    # values = sorted(edge_dict.items(), key=lambda x: x[1], reverse=True)

    edge_w = np.ones(num_e1)
    for i in range(num_e1):
        edge = (edge_index1[0, i], edge_index1[1, i])
        edge_w[i] = math.exp(- lam * edge_dict[edge])

    values = sorted(edge_dict.items(), key=lambda x: x[1], reverse=True)
    # print(values)
    import matplotlib.pyplot as plt
    plt.hist(x=edge_dict.values(), bins=20, color="steelblue", edgecolor="black")
    plt.show()
    values0 = np.array([i[0] for i in values if i[1] == 0]).T
    values1 = np.array([i[0] for i in values if i[1] > 0]).T
    values2 = values1[:, values1[0] < values1[1]]
    values3 = values2[:, int(values2.shape[1] / 4):]
    edges = np.concatenate([values3, values3[[1, 0], :], values0], axis=1)
    edges = torch.tensor(edges)

    return edge_w
