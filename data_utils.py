import copy
import os
import dgl
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import pickle
import ray
from tqdm import tqdm

def get_feats(G):
    if isinstance(G, dgl.DGLGraph) or isinstance(G, dgl.DGLHeteroGraph):
        initial_num_nodes = G.number_of_nodes()

        if 'attr' in G.ndata.keys():
            node_feats = G.ndata['attr']
        else:
            node_feats = torch.ones(G.number_of_nodes(), 1)

        if 'attr' in G.edata.keys():
            edge_feats = G.edata['attr']
        else:
            edge_feats = None

        # G = dgl.to_networkx(G).to_undirected()
        G = dgl.to_networkx(G.cpu()).to_undirected()
        assert G.number_of_nodes() == initial_num_nodes, f'mismatch in number of nodes, {initial_num_nodes},' \
                                                         f' {G.number_of_nodes()}'
    else:
        node_feats = torch.ones(G.number_of_nodes(), 1)
        edge_feats = None

    return G, node_feats, edge_feats


def calculate_all_structural_feats(G):
    G_simple = nx.Graph(G)  # clustering coefficient is not defined for multi-graphs/directed graphs
    feats = {'d': torch.tensor(list(G.degree))[:, 1].view(-1, 1).float(),
             'c': torch.Tensor(list(nx.clustering(G_simple).values())).view(-1, 1).float(),
             'o': torch.Tensor(list(nx.square_clustering(G_simple).values())).view(-1, 1).float()
             }
    return feats


def add_given_structural_feats(nx_g, G, deg, clus, orbit):
    G_simple = nx.Graph(nx_g)  # clustering coefficient is not defined for multi-graphs/directed graphs
    new_x = [G.x]
    device = G.x.device
    if deg:
        new_x.append(torch.tensor(list(nx_g.degree))[:, 1].view(-1, 1).float().to(device))
    if clus:
        new_x.append(torch.Tensor(list(nx.clustering(G_simple).values())).view(-1, 1).float().to(device))
    if orbit:
        new_x.append(torch.Tensor(list(nx.square_clustering(G_simple).values())).view(-1, 1).float().to(device))

    if len(new_x) > 1:
        new_x = torch.cat(new_x, dim=1)
        G.x = new_x

    G.added_struct_feats = True


def standardize_graph(G, args):
    device = args['device']
    if not isinstance(G, Data):
        nx_g, node_feats, edge_feats = get_feats(G)
        G = from_networkx(nx_g)
        G.x = node_feats
        G.edge_attr = edge_feats
        if G.x is not None:
            G.x = G.x.to(device)
        if G.edge_attr is not None:
            G.edge_attr = G.edge_attr.to(device)
    else:
        nx_g = to_networkx(G)
    add_given_structural_feats(nx_g, G, args['d'], args['c'], args['o'])
    return G


remote_standardize_graph = ray.remote(standardize_graph)


def standardize_dataset_for_gnn(dataset, args):
    if hasattr(dataset[0], 'added_struct_feats') and dataset[0].added_struct_feats:
        return dataset
    # copying to make sure not making change in the previous dataset:
    args['device'] = dataset[0].device
    new_dataset = copy.deepcopy(dataset)
    if args['is_parallel']:
        # parallel version:
        result_ids = []
        for sample in new_dataset:
            result_ids.append(remote_standardize_graph.remote(sample, args))
        return ray.get(result_ids)
    else:
        # non-parallel version:
        results = []
        for sample in new_dataset:
            results.append(standardize_graph(sample, args))
        return results


def make_graph_ready_to_save_format(G):
    G, node_feats, edge_feats = get_feats(G)  # dgl to nx graph
    structural_feats = calculate_all_structural_feats(G)

    G = from_networkx(G) # nx to pyg
    G.x = node_feats
    G.edge_attr = edge_feats
    G.d = structural_feats['d']
    G.c = structural_feats['c']
    G.o = structural_feats['o']

    return G


remote_make_graph_ready_to_save_format = ray.remote(make_graph_ready_to_save_format)


def make_dataset_ready_to_save(dataset, parallel=False):
    new_dataset = copy.deepcopy(dataset)
    if parallel:
        # parallel version:
        result_ids = []
        for sample in new_dataset:
            result_ids.append(remote_make_graph_ready_to_save_format.remote(sample))
        return ray.get(result_ids)
    else:
        # non-parallel version:
        results = []
        for sample in tqdm(new_dataset, desc='dgl to pyg: series'):
            results.append(make_graph_ready_to_save_format(sample))
        return results


def fix_features_graph(G, deg, clus, orbit):
    if hasattr(G, 'added_struct_feats') and G.added_struct_feats:
        print('Aborting, G already has structured features added')
    feats = [G.x]
    # if deg:
    #     feats.append(G.d)
    # if clus:
    #     feats.append(G.c)
    # if orbit:
    #     feats.append(G.o)
    device = G.x.device
    if deg:
        feats.append(G.d.to(device))
    if clus:
        feats.append(G.c.to(device))
    if orbit:
        feats.append(G.o.to(device))

    G.x = torch.cat(feats, dim=1)
    G.added_struct_feats = True
    assert G.x.shape[0] == G.num_nodes, f'mismatch in num nodes:{G.num_nodes}, feature shape dim 1:{G.x.shape[0]}'


def make_dataset_from_saved_format(data, deg, clus, orbit):
    for G in data:
        fix_features_graph(G, deg, clus, orbit)


def load_and_prep_data(args):
    drc = args.permute_data_drc
    try:
        ref_set = pickle.load(open(drc + '/reference_set', 'rb'))

        permuted_sets_names = []
        for root, subdir, files in os.walk(drc):
            for file in files:
                if file.startswith('permuted_set_'):
                    permuted_sets_names.append(file)

        permuted_sets_names.sort()
        permuted_sets = []
        for name in permuted_sets_names:
            with open(drc + '/' + name, 'rb') as file:
                permuted_sets.append(pickle.load(file))

        if not hasattr(ref_set[0], 'added_struct_feats'):
            make_dataset_from_saved_format(ref_set, args.deg_feats, args.clus_feats, args.orbit_feats)
            for permuted_set in permuted_sets:
                make_dataset_from_saved_format(permuted_set, args.deg_feats, args.clus_feats, args.orbit_feats)

        return ref_set, permuted_sets
    except:
        return None, None
