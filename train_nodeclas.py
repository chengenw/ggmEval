import os.path as osp
import time
import argparse

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

# custom modules
from maskgae.utils import Logger, set_seed, tab_printer
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath

from tqdm import tqdm
from termcolor import cprint
from copy import deepcopy
from torch_geometric.utils import (to_undirected, to_dense_adj, dense_to_sparse)
def train_maskgae(args, graphs): # graphs: pyg dataset proteins etc.
    device = args.device

    all_data = []
    for data in tqdm(graphs, desc='graphs'):
        assert data.is_undirected()
        # data_split = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
        data_split = T.RandomLinkSplit(num_val=0.15, num_test=0.15,
                                                            is_undirected=True,
                                                            split_labels=True,
                                                            add_negative_train_samples=False)(data)
        all_data.append(data_split)
    train_data, val_data, test_data = zip(*all_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    splits = dict(train=train_loader, valid=val_loader, test=test_loader)

    if args.mask == 'Path':
        mask = MaskPath(p=args.p,
                        start=args.start,
                        walk_length=3)
    elif args.mask == 'Edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = None  # vanilla GAE

    encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                   num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

    # print(model)

    def train(data_loader):
        model.train()
        loss = model.train_epoch_loader(data_loader, optimizer,
                                 alpha=args.alpha, edge_batch_size=args.edge_batch_size, device=args.device)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        train_loader = splits['train']
        # z = model.forward_loader(train_loader, device)

        # cprint(f'\n*** check if nodes order remains the same\n', 'red')
        valid_auc, valid_ap = model.test_loader(train_loader, splits['valid'], device)
        test_auc, test_ap = model.test_loader(train_loader, splits['test'], device)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    save_path = args.save_path
    args.model_name = f'{args.dataset}_maskgae_checkpoint.pt'
    model_dir = f'saved_models'
    model_path =f'{model_dir}/{args.model_name}'
    runs = 1
    loggers = {
        'AUC': Logger(runs, args),
        'AP': Logger(runs, args),
    }
    print('Start Training (Link Prediction Pretext Training)...')
    for run in range(runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        pbar = tqdm(range(1, 1 + args.epochs))
        for epoch in pbar:

            t1 = time.time()
            loss = train(splits['train'])
            t2 = time.time()

            pbar.set_postfix({'loss': loss})
            pbar.update()

            if epoch % args.eval_period == 0:
                results = test(splits)

                valid_result = results[monitor][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    torch.save(model, model_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                for key, result in results.items():
                    valid_result, test_result = result
                    print(key)
                    print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                          f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                          f'Best_epoch: {best_epoch:02d}, '
                          f'Best_valid: {best_valid:.2%}%, '
                          f'Loss: {loss:.4f}, '
                          f'Valid: {valid_result:.2%}, '
                          f'Test: {test_result:.2%}',
                          f'Training Time/epoch: {t2-t1:.3f}')
                print('#' * round(140*epoch/(args.epochs+1)))
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(save_path))
        results = test(splits, model)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result (Link Prediction Pretext Training)')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
#
# def train_linkpred(model, splits, args, device="cpu"):
#
#     def train(data):
#         model.train()
#         loss = model.train_epoch(data.to(device), optimizer,
#                                  alpha=args.alpha, batch_size=args.batch_size)
#         return loss
#
#     @torch.no_grad()
#     def test(splits, batch_size=2**16):
#         model.eval()
#         train_data = splits['train'].to(device)
#         z = model(train_data.x, train_data.edge_index)
#
#         valid_auc, valid_ap = model.test(
#             z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size)
#
#         test_auc, test_ap = model.test(
#             z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size)
#
#         results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
#         return results
#
#     monitor = 'AUC'
#     save_path = args.save_path
#     runs = 1
#     loggers = {
#         'AUC': Logger(runs, args),
#         'AP': Logger(runs, args),
#     }
#     print('Start Training (Link Prediction Pretext Training)...')
#     for run in range(runs):
#         model.reset_parameters()
#
#         optimizer = torch.optim.Adam(model.parameters(),
#                                      lr=args.lr,
#                                      weight_decay=args.weight_decay)
#
#         best_valid = 0.0
#         best_epoch = 0
#         cnt_wait = 0
#         for epoch in range(1, 1 + args.epochs):
#
#             t1 = time.time()
#             loss = train(splits['train'])
#             t2 = time.time()
#
#             if epoch % args.eval_period == 0:
#                 results = test(splits)
#
#                 valid_result = results[monitor][0]
#                 if valid_result > best_valid:
#                     best_valid = valid_result
#                     best_epoch = epoch
#                     torch.save(model.state_dict(), save_path)
#                     cnt_wait = 0
#                 else:
#                     cnt_wait += 1
#
#                 for key, result in results.items():
#                     valid_result, test_result = result
#                     print(key)
#                     print(f'Run: {run + 1:02d} / {args.runs:02d}, '
#                           f'Epoch: {epoch:02d} / {args.epochs:02d}, '
#                           f'Best_epoch: {best_epoch:02d}, '
#                           f'Best_valid: {best_valid:.2%}%, '
#                           f'Loss: {loss:.4f}, '
#                           f'Valid: {valid_result:.2%}, '
#                           f'Test: {test_result:.2%}',
#                           f'Training Time/epoch: {t2-t1:.3f}')
#                 print('#' * round(140*epoch/(args.epochs+1)))
#                 if cnt_wait == args.patience:
#                     print('Early stopping!')
#                     break
#         print('##### Testing on {}/{}'.format(run + 1, args.runs))
#
#         model.load_state_dict(torch.load(save_path))
#         results = test(splits, model)
#
#         for key, result in results.items():
#             valid_result, test_result = result
#             print(key)
#             print(f'**** Testing on Run: {run + 1:02d}, '
#                   f'Best Epoch: {best_epoch:02d}, '
#                   f'Valid: {valid_result:.2%}, '
#                   f'Test: {test_result:.2%}')
#
#         for key, result in results.items():
#             loggers[key].add_result(run, result)
#
#     print('##### Final Testing result (Link Prediction Pretext Training)')
#     for key in loggers.keys():
#         print(key)
#         loggers[key].print_statistics()
#
#
# def train_nodeclas(model, data, args, device='cpu'):
#     def train(loader):
#         clf.train()
#         for nodes in loader:
#             optimizer.zero_grad()
#             loss_fn(clf(embedding[nodes]), y[nodes]).backward()
#             optimizer.step()
#
#     @torch.no_grad()
#     def test(loader):
#         clf.eval()
#         logits = []
#         labels = []
#         for nodes in loader:
#             logits.append(clf(embedding[nodes]))
#             labels.append(y[nodes])
#         logits = torch.cat(logits, dim=0).cpu()
#         labels = torch.cat(labels, dim=0).cpu()
#         logits = logits.argmax(1)
#         return (logits == labels).float().mean().item()
#
#     if hasattr(data, 'train_mask'):
#         train_loader = DataLoader(data.train_mask.nonzero().squeeze(), pin_memory=False, batch_size=512, shuffle=True)
#         test_loader = DataLoader(data.test_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
#         val_loader = DataLoader(data.val_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
#     else:
#         train_loader = DataLoader(data.train_nodes.squeeze(), pin_memory=False, batch_size=4096, shuffle=True)
#         test_loader = DataLoader(data.test_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
#         val_loader = DataLoader(data.val_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
#
#     data = data.to(device)
#     y = data.y.squeeze()
#     embedding = model.encoder.get_embedding(data.x, data.edge_index, l2_normalize=args.l2_normalize)
#
#     loss_fn = nn.CrossEntropyLoss()
#     clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)
#
#     logger = Logger(args.runs, args)
#
#     print('Start Training (Node Classification)...')
#     for run in range(args.runs):
#         nn.init.xavier_uniform_(clf.weight.data)
#         nn.init.zeros_(clf.bias.data)
#         optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=args.nodeclas_weight_decay)  # 1 for citeseer
#
#         best_val_metric = test_metric = 0
#         start = time.time()
#         for epoch in range(1, 101):
#             train(train_loader)
#             val_metric, test_metric = test(val_loader), test(test_loader)
#             if val_metric > best_val_metric:
#                 best_val_metric = val_metric
#                 best_test_metric = test_metric
#             end = time.time()
#             if args.debug:
#                 print(f"Epoch {epoch:02d} / {100:02d}, Valid: {val_metric:.2%}, Test {test_metric:.2%}, Best {best_test_metric:.2%}, Time elapsed {end-start:.4f}")
#
#         print(f"Run {run+1}: Best test accuray {best_test_metric:.2%}.")
#         logger.add_result(run, (best_val_metric, best_test_metric))
#
#     print('##### Final Testing result (Node Classification)')
#     logger.print_statistics()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
    parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

    parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
    parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
    parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
    parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size for link prediction training. (default: 2**16)')

    parser.add_argument("--start", nargs="?", default="node", help="Which Type to sample starting nodes for random walks, (default: node)")
    parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
    parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
    parser.add_argument("--save_path", nargs="?", default="model_nodeclas", help="save path for model. (default: model_nodeclas)")
    parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--edge_batch_size', type=int, default=2**9)
    parser.add_argument('--num_copy', type=int, default=3)


    try:
        args = parser.parse_args()
        print(tab_printer(args))
    except:
        parser.print_help()
        exit(0)

    if not args.save_path.endswith('.pth'):
        args.save_path += '.pth'

    set_seed(args.seed)
    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])


    # root = '~/public_data/pyg_data' # my root directory
    root = 'data/'

    if args.dataset in {'arxiv', 'products', 'mag'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{args.dataset}')
        if args.dataset in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_nodes = split_idx['train']['paper']
            data.val_nodes = split_idx['valid']['paper']
            data.test_nodes = split_idx['test']['paper']
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_nodes = split_idx['train']
            data.val_nodes = split_idx['valid']
            data.test_nodes = split_idx['test']

    elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, args.dataset)
        data = transform(dataset[0])

    elif args.dataset == 'Reddit':
        dataset = Reddit(osp.join(root, args.dataset))
        data = transform(dataset[0])
    elif args.dataset in {'Photo', 'Computers'}:
        dataset = Amazon(root, args.dataset)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif args.dataset in {'CS', 'Physics'}:
        dataset = Coauthor(root, args.dataset)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    else:
        raise ValueError(args.dataset)

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=False)(data)

    splits = dict(train=train_data, valid=val_data, test=test_data)


    if args.mask == 'Path':
        mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                        start=args.start,
                        walk_length=3)
    elif args.mask == 'Edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = None # vanilla GAE

    encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                   num_layers=args.decoder_layers, dropout=args.decoder_dropout)


    model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

    # print(model)
    #
    # train_linkpred(model, splits, args, device=device)
    # train_nodeclas(model, data, args, device=device)

    graphs = []
    for _ in range(args.num_copy):
        perm = torch.randperm(data.x.shape[0])
        x = data.x[perm]
        # edge_index = data.edge_index
        adj = to_dense_adj(to_undirected(data.edge_index)).squeeze(0)
        adj = adj[perm][:, perm]
        # adj = adj.triu()
        edge_index = dense_to_sparse(adj)[0]
        graphs.append(Data(x, edge_index))
        # graphs.append(deepcopy(data))
    # graphs = [deepcopy(data) for _ in range(args.num_copy)]
    train_maskgae(args, graphs)

if __name__ == '__main__':
    main()