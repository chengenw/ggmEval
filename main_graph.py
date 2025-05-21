import logging
import os.path

import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from time import strftime

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model

from termcolor import cprint

def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False, embed_only=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, batch_g in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.x
            labels = batch_g.y.cpu()
            out = model.embed(feat, batch_g.edge_index, batch_g.edge_attr)
            if pooler == "mean":
                out = global_mean_pool(out, batch_g.batch)
            elif pooler == "max":
                out = global_max_pool(out, batch_g.batch)
            elif pooler == "sum":
                out = global_add_pool(out, batch_g.batch)
            else:
                raise NotImplementedError

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    if embed_only:
        return x

    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_acc: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in tqdm(kf.split(embeddings, labels)):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        acc = (y_test == preds).sum() / len(y_test)
        # assert f1 == acc, print(f'{f1}\n{acc}\n{np.array(y_test)}\n{np.array(preds)}')
        # if f1 != acc:
        #     cprint(f'\n{f1}\n{acc}\n{np.array(y_test)}\n{np.array(preds)}', 'red')
        # result.append(f1)
        result.append(acc)
    test_acc = np.mean(result)
    test_std = np.std(result)

    return test_acc, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None, writer=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch), desc='graphmae pretrain')
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)

            feat = batch_g.x
            model.train()
            loss, loss_dict = model(feat, batch_g.edge_index, batch_g.edge_attr, batch=batch_g.batch)
            # if 'edge_attr' in batch_g:
            #     loss, loss_dict = model(feat, batch_g.edge_index, batch_g.edge_attr)
            # else:
            #     loss, loss_dict = model(feat, batch_g.edge_index)

            k, v = list(loss_dict.items())[0]
            writer.add_scalar(k, v, epoch)
            # writer.add_scalars('loss', {k:v}, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

def train_mae(args, graphs, labels=None):
    device = args.device
    # cprint(f'args.device is {device}, current device is {torch.cuda.current_device()}', 'red')
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooler = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    args.num_features = graphs[0].x.shape[1]
    cprint(f'dataset_name {dataset_name}: Num Graphs {len(graphs)}, Num Feat {args.num_features}', 'yellow')
    if args.use_edge_attr and 'edge_attr' in graphs[0] and dataset_name in ['zinc', 'ZINC']:
        args.edge_dim = graphs[0].edge_attr.shape[1]
    else:
        # assert args.edge_dim is None
        cprint(f'args.edge_dim is {args.edge_dim}', 'red')

    all_labels = {}
    if labels is not None:
        for i, g in enumerate(graphs):
            g.y = torch.tensor([labels[i]])
            all_labels[labels[i]] = -1
        cprint(f'Num Classes: {len(all_labels.keys())}', 'yellow')

    train_loader = DataLoader(graphs, batch_size=batch_size, pin_memory=False)  # pin_memory
    eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    seed = args.seed
    # for i, seed in enumerate(seeds):
    cprint(f"####### Run for seed {seed}", 'red')
    set_random_seed(seed)

    if logs:
        logger = TBLogger(
            name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
    else:
        logger = None

    model = build_model(args)
    model.to(device)
    model.pooler = args.pooling
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    if not load_model:
        model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler,
                         None, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger, args.writer)
        model = model.cpu()

    model_dir = f'saved_models'
    model_name = f'{args.dataset}_mae_checkpoint.pt'
    args.model_name = model_name
    model_name_d = f'{model_dir}/{model_name}'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if load_model:
        logging.info("Loading Model ... ")
        # model.load_state_dict(torch.load(model_name_d))
        model = torch.load(model_name_d)
    if not load_model and save_model:
        logging.info("Saveing Model ...")
        torch.save(model.state_dict(), f"{model_dir}/{args.dataset}_checkpoint.pt")
        torch.save(model, model_name_d)
    else:
        cprint(f'@@@GraphMAE model not saved!', 'red')

    if labels is not None:
        model = model.to(device)
        model.eval()
        graph_classification_evaluation(model, pooler, eval_loader, -1, lr_f, weight_decay_f,
                                                  max_epoch_f, device, mute=False)


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooler = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    args.num_features = num_features
    if 'edge_attr' in graphs[0]:
        args.edge_dim = graphs[0].edge_attr.shape[1]

    train_idx = torch.arange(len(graphs))
    
    train_loader = DataLoader(graphs, batch_size=batch_size, pin_memory=True)
    eval_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            model = model.cpu()

        model_dir = f'save_models'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load(f"{model_dir}/{args.dataset}_checkpoint.pt"))
        if not load_model and save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), f"{model_dir}/{args.dataset}_checkpoint.pt")
        
        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(model, pooler, eval_loader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False)
        acc_list.append(test_f1)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)