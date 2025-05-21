
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# export CUDA_VISIBLE_DEVICES=1

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'device is {device}, current device is {torch.cuda.current_device()}')

import dgl
import numpy as np

from utils import experiment_logging as helper
from evaluation.evaluator import Evaluator
import traceback
import logging
from permuters import SampleSizePermuter, MixingPermuter, RewiringPermuter, ModePermuter, ComputationEffPermuter, RandomFeatPermuter
import utils.graph_generators as generators
import warnings
import time
from config import Config
from pathlib import Path
from GIN_train_pyg import get_model
import random
from tqdm import tqdm
import torch.nn.functional as F

from termcolor import cprint
import pickle
import re

warnings.filterwarnings('ignore')

dataset_names = ['PROTEINS', 'MUTAG', 'DD', 'DBLP_v1']
dataset_social = ['IMDB-MULTI', 'IMDB-BINARY', 'reddit_threads', 'REDDIT-BINARY', 'github_stargazers', 'TWITTER-Real-Graph-Partial', 'REDDIT-MULTI-5K', 'COLLAB', 'deezer_ego_nets', 'twitch_egos', 'REDDIT-MULTI-12K']
dataset_1 = ['lobster', 'grid', 'community', 'ego']
def label2attr(dataset):
    feature_dim = 0
    for g in dataset:
        feature_dim = max(feature_dim, g.ndata["label"].max().item())

    feature_dim += 1
    for g in dataset:
        node_label = g.ndata["label"].view(-1)
        feat = F.one_hot(node_label, num_classes=feature_dim).float()
        g.ndata["attr"] = feat

def generate_dataset(args, device):
    """Generate (or load) a given dataset.

    Parameters
    ----------
    args : Argparse dict
        The command-line args parsed by Argparse
    device : torch.device
        The device to move the generated graphs to.

    Returns
    -------
    List of DGL graphs
        The generated (or loaded) dataset moved to the
        specified device.

    """
    dataset_name = args.dataset

    seed = args.seed
    if dataset_name == 'grid':
        reference_dset = generators.make_grid_graphs()
    elif dataset_name == 'lobster':
        reference_dset = generators.make_lobster_graphs()
    elif dataset_name == 'er':  # erdos-renyi
        args.er_p = np.random.uniform(low=0.05, high=0.95)
        reference_dset = generators.make_er_graphs(seed, args.er_p)
    elif dataset_name in dataset_names or dataset_name in dataset_social:
        reference_dset, labels = generators.load_proteins(node_attributes=args.node_attr, dataset_name=dataset_name) # node_attr not defined
    elif dataset_name == 'lego':
        reference_dset = generators.load_lego()
    elif dataset_name == 'community':
        reference_dset = generators.make_community_graphs()
    elif dataset_name == 'community-large':
        reference_dset = generators.make_community_graphs_large()
    elif dataset_name == 'ego':
        reference_dset = generators.make_ego_graphs()
    elif dataset_name == 'zinc':
        reference_dset = generators.load_zinc()
    elif dataset_name == 'zinc-large':
        reference_dset = generators.load_zinc_large()
    elif dataset_name == 'cifar10':
        reference_dset = generators.load_cifar10()
    # elif dataset_name.lower() in ['mnist', 'mnist_1k', 'mnist_3k', 'pattern', 'pattern_1k', 'pattern_3k']:
    elif re.match(r'(mnist|pattern|cifar10|tsp)(_\d+k)?', dataset_name):
        with open(f'data/graphs/datasets/{dataset_name.lower()}.pkl', 'rb') as f:
            reference_dset = pickle.load(f)
    else:
        raise Exception(dataset_name)

    print('Dataset size:', len(reference_dset))
    num_nodes = [g.number_of_nodes() for g in reference_dset]
    num_edges = [g.number_of_edges() for g in reference_dset]
    print('Min num nodes:', np.min(num_nodes))
    print('Max num nodes:', np.max(num_nodes))
    print('Mean num nodes:', np.mean(num_nodes))
    print('Min num edges:', np.min(num_edges))
    print('Max num edges:', np.max(num_edges))
    print('Mean num edges:', np.mean(num_edges))

    if not isinstance(reference_dset[0], dgl.DGLGraph):
        # reference_dset = [dgl.DGLGraph(g) for g in tqdm(reference_dset, 'nx to dgl')]
        ref_d = []
        cond = dataset_name in dataset_names and args.node_label
        node_attrs = ['label'] if cond else None
        for g in tqdm(reference_dset, 'nx to dgl'):
            g_ = dgl.from_networkx(g, node_attrs=node_attrs)
            ref_d.append(g_)
        if cond:
            cprint(f'args.node_label is True', 'red')
            label2attr(ref_d)
        else:
            cprint(f'args.node_label is False', 'red')
        reference_dset = ref_d

    reference_dset = [g.to(device) for g in reference_dset]

    if dataset_name in dataset_names:
        reference_dset = (reference_dset, labels)

    return reference_dset


def get_graph_permuter(reference_set, helper, evaluator):
    """Initialize the experiment.

    Parameters
    ----------
    helper : helper.ExperimentHelper
        General experiment helper --- logging results etc.
    evaluator : Evaluator
        The evaluator object used to compute each metric.

    Returns
    -------
    BasePermuter
        The graph permuter that alters the graphs according
        to the specified experiments and computes metrics.

    Args:
        reference_set:

    """
    args = helper.args
    permutation_type = args.permutation_type

    if permutation_type == 'sample-size-random':
        return SampleSizePermuter.SampleSizePermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mixing-gen'\
            or permutation_type == 'mixing-random':
        return MixingPermuter.MixingPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'rewiring-edges':
        return RewiringPermuter.RewiringPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mode-collapse':
        return ModePermuter.ModeCollapsePermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mode-dropping':
        return ModePermuter.ModeDroppingPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'computation-eff-size':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='size')

    elif permutation_type == 'computation-eff-qty':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='qty')

    elif permutation_type == 'computation-eff-edges':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='edges')

    elif 'randomize' in permutation_type:
        return RandomFeatPermuter.RandomFeatPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)
    else:
        raise Exception('not implemented')


if __name__ == '__main__':
    # Fetch the command line arguments
    config = Config().parse()
    config.dataset_names = dataset_names

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = False

    # For logging results mostly
    helper = helper.ExperimentHelper(
        config, results_dir=config.results_directory)

    try:
        if helper.args.no_cuda or not torch.cuda.is_available():
            helper.args.no_cuda = True
            helper.args.device = torch.device('cpu')
        else:
            helper.args.device = torch.device('cuda')

        reference_set = generate_dataset(helper.args, device=helper.args.device)
        labels = None
        dataset_name = helper.args.dataset
        if dataset_name in dataset_names:
            reference_set, labels = reference_set

        _ref = list(enumerate(reference_set))
        random.shuffle(_ref)
        indices, reference_set = zip(*_ref)
        reference_set = list(reference_set)
        if dataset_name in dataset_names:
            labels = np.array(labels)[list(indices)]

        if helper.args.split:
            split_point = int(len(reference_set) * helper.args.split_ratio)
            train_set = reference_set[:split_point]
            test_set = reference_set[split_point:]
        else:
            train_set = reference_set
            test_set = reference_set

        path = Path(f'saved_models/{helper.args.model_name}')
        if helper.args.feature_extractor == 'gin-random' or helper.args.retrain or not path.is_file():
            get_model(train_set, helper.args, labels)

        feature_adder_args = {'d': helper.args.deg_feats,
                              'c': helper.args.clus_feats,
                              'o': helper.args.orbit_feats,
                              'is_parallel': helper.args.is_parallel}
        # Get object for computing desired metrics
        helper.args.feature_adder_args = feature_adder_args
        evaluator = Evaluator(**helper.args)

        # Get object to apply appropriate permutations to graphs
        graph_permuter = get_graph_permuter(test_set, helper, evaluator)

        start = time.time()
        res = graph_permuter.perform_run()
        total = time.time() - start
        total = total / 60
        helper.logger.info('EXPERIMENT TIME: {} mins'.format(total))

    except:
        graph_permuter.save_results_final()
        traceback.print_exc()
        logging.exception('')

    finally:
        helper.end_experiment()
