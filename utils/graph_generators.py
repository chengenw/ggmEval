import dgl
import networkx as nx
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import shutil
# import dgl
import random

def make_grid_graphs(lower=10, upper=20, **kwargs):
    graphs = []
    for i in range(lower, upper):
        for j in range(lower, upper):
            graphs.append(nx.grid_2d_graph(i, j))

    return graphs

def make_lobster_graphs(seed=1234, mean_num_nodes=80, num_graphs=100, **kwargs):
    path = 'data/graphs/datasets/lobster.h5'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    # Same as GRAN
    graphs = []
    p1 = 0.7
    p2 = 0.7
    min_node = 10
    max_node = 100
    tmp_seed = seed

    while len(graphs) < num_graphs:
        g = nx.random_lobster(mean_num_nodes, p1, p2, seed=tmp_seed)
        if g.number_of_nodes() >= min_node and g.number_of_nodes() <= max_node:
            graphs.append(g)
        tmp_seed += 1
    pickle.dump(graphs, open(path, 'wb'))
    return graphs
    # return [ for _ in range(num_graphs)]

def make_community_graph(c_sizes, g_p=0.3, p_inter=0.05):#, p_inter=0.01):
    graphs = [nx.fast_gnp_random_graph(c_size, g_p, seed=np.random.choice(1000)) for c_size in c_sizes]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for ix, community1 in enumerate(communities):
        nodes1 = list(community1.nodes())
        for community2 in communities[ix + 1:]:
            nodes2 = list(community2.nodes())

            num_inter_edges = int((len(nodes1) + len(nodes2)) * p_inter)
            # print(num_inter_edges, community1.number_of_edges(), community2.number_of_edges())
            edges1 = np.random.choice(nodes1, size=num_inter_edges)
            edges2 = np.random.choice(nodes2, size=num_inter_edges)
            G.add_edges_from(zip(edges1, edges2))
    return G

def make_community_graphs():
    path = 'data/graphs/datasets/community.h5'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))

    graphs = []
    num_communities = 2
    for k in range(500):
        c_sizes = [np.random.choice(list(range(30, 80)))]
        graphs.append(make_community_graph(c_sizes * num_communities))

    pickle.dump(graphs, open(path, 'wb'))
    return graphs

def make_community_graphs_large():
    path = 'data/graphs/datasets/community_large.h5'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))

    graphs = []
    num_communities = 10
    for k in range(400):
        c_sizes = np.random.choice(list(range(100, 200)), size=num_communities)
        graphs.append(make_community_graph(c_sizes, g_p=0.15, p_inter=0.025))

    pickle.dump(graphs, open(path, 'wb'))
    return graphs

def make_ego_graphs():
    import scipy.sparse as sp
    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def Graph_load(dataset = 'cora'):
        '''
        Load a single graph dataset
        :param dataset: dataset name
        :return:
        '''
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            load = pickle.load(open("data/graphs/datasets/{}/ind.{}.{}".format(dataset, dataset, names[i]), 'rb'), encoding='latin1')
            # print('loaded')
            objects.append(load)
            # print(load)
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/graphs/datasets/{}/ind.{}.test.index".format(dataset, dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        G = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(G)
        return adj, features, G

    _, _, G = Graph_load(dataset='citeseer')
    subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
    G = max(subgraphs, key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    return graphs

def make_er_graphs(seed, p, lower=5, upper=55, **kwargs):
    if kwargs.get('num_graphs') is None: # Generate based on the lower/upper range
        graphs = []
        for num_nodes in range(lower, upper):
            graphs.append(nx.erdos_renyi_graph(num_nodes, p, seed=seed))
            graphs.append(nx.erdos_renyi_graph(num_nodes, p, seed=seed + 1))
        return graphs
    elif kwargs.get('num_graphs') is not None: # Generate based on num_graphs and given size
        return [nx.erdos_renyi_graph(kwargs.get('mean_num_nodes'), p, seed=seed) for _ in range(kwargs.get('num_graphs'))]


def load_graphs(name, min_num_nodes=20,
                  max_num_nodes=500,
                  node_attributes=False,
                  graph_labels=False, p=1.0):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    data_dir = 'data/graphs/datasets'
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        path_ = path + '_'
        dataset = TUDataset(path_, name=name)
        shutil.move(f'{path_}/{name}/raw', f'{data_dir}')
        os.rename(f'{data_dir}/raw', f'{data_dir}/{name}')
        shutil.rmtree(path_)
    data_adj = np.loadtxt(
        os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
    if node_attributes:
      data_node_att = np.loadtxt(
          os.path.join(path, '{}_node_attributes.txt'.format(name)),
          delimiter=',')

    data_graph_indicator = np.loadtxt(
        os.path.join(path, '{}_graph_indicator.txt'.format(name)),
        delimiter=',').astype(int)
    if graph_labels:
      data_graph_labels = np.loadtxt(
          os.path.join(path, '{}_graph_labels.txt'.format(name)),
          delimiter=',').astype(int)

    if os.path.exists(f'{path}/{name}_node_labels.txt'):
        data_node_label = np.loadtxt(
        os.path.join(path, '{}_node_labels.txt'.format(name)),
        delimiter=',').astype(int)
    else:
        data_node_label = np.zeros((data_graph_indicator.shape), dtype=np.int64)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
      if node_attributes:
        G.add_node(i + 1, feature=data_node_att[i])
      G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    labels = []
    for i in range(graph_num):
      cur_p = np.random.uniform()
      if cur_p > p:  # sample graph with probability p
        continue
      # find the nodes for each graph
      nodes = node_list[data_graph_indicator == i + 1]
      G_sub = G.subgraph(nodes)
      if graph_labels:
        G_sub.graph['label'] = data_graph_labels[i]
      # print('nodes', G_sub.number_of_nodes())
      # print('edges', G_sub.number_of_edges())
      # print('label', G_sub.graph)
      if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
      ) <= max_num_nodes and G_sub.number_of_edges() >= 1:
        labels.append(data_graph_labels[i])
        graphs.append(G_sub)
        if G_sub.number_of_nodes() > max_nodes:
          max_nodes = G_sub.number_of_nodes()
        # print(G_sub.number_of_nodes(), 'i', i)
        # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
        # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))

    # res = dgl.from_networkx(graphs[0], node_attrs=['label'])
    return graphs, labels

def load_proteins(min_num_nodes=3,
                  max_num_nodes=1000,
                  node_attributes=False,
                  graph_labels=True, dataset_name='DD', p=1.0):
    # at least 3 nodes for rewiring permutation

    # if dataset_name == 'mutag':
    #     dataset_name = dataset_name.upper()
    #     min_num_nodes, max_num_nodes = 0, 10000
    # if dataset_name == 'proteins':
    #     dataset_name = 'DD'
    # elif dataset_name == 'pp':
    #     min_num_nodes, max_num_nodes = 10, 10000
    #     dataset_name = 'Proteins'
    # else:
    #     min_num_nodes, max_num_nodes = 0, 10000

    if dataset_name == 'PROTEINS' or dataset_name == 'MUTAG':
        min_num_nodes = 20
    if dataset_name == 'DD':
        min_num_nodes, max_num_nodes = 100, 500

    if dataset_name == 'DBLP_v1':
        # p = 0.1 # 19456->3003 (min_node=20), 3003-> 313 (p=0.1)
        # p = 0.02 400 graphs min_num_nodes=0
        p = 0.04
    elif dataset_name == 'IMDB-MULTI' or dataset_name == 'IMDB-BINARY':
        p = 1.0
    elif dataset_name == 'reddit_threads':
        p = 0.004
    elif dataset_name == 'REDDIT-BINARY':
        p = 0.4
    elif dataset_name == 'github_stargazers':
        p = 0.07
    elif dataset_name == 'TWITTER-Real-Graph-Partial':
        p = 0.006
    elif dataset_name == 'REDDIT-MULTI-5K':
        p = 0.16
    elif dataset_name == 'COLLAB':
        p = 0.16
    elif dataset_name == 'deezer_ego_nets':
        p = 0.1
    elif dataset_name == 'twitch_egos':
        p = 0.007
    elif dataset_name == 'REDDIT-MULTI-12K':
        p = 0.07


    # dataset_name = dataset_name.upper()

    return load_graphs(dataset_name, min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
            node_attributes=node_attributes, graph_labels=graph_labels, p=p)

def load_firstmm_db(min_num_nodes=0,
        max_num_nodes=10000,
        node_attributes=False,
        graph_labels=False):
    return load_graphs('FIRSTMM_DB', min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
            node_attributes=node_attributes, graph_labels=graph_labels)


def load_zinc():
    return pickle.load(open('data/graphs/datasets/zinc.h5', 'rb'))[: 1000]
