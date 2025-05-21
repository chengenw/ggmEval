import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Experiment args')
        subparsers = self.parser.add_subparsers()

        self.parser.add_argument(
            '--no_cuda', action='store_true',
            help='Flag to disable cuda')

        self.parser.add_argument(
            '--seed', default=42, type=int,
            help='the random seed to use')

        self.parser.add_argument(
            '--dataset', default='lobster',
            # choices=['grid', 'lobster', 'lego', 'proteins', 'community', 'ego', 'zinc', 'mutag', 'pp'],
            help='The dataset to use')

        self.parser.add_argument(
            '--permutation_type', default='mode-collapse',
            choices=['sample-size-random', 'mixing-gen', 'mixing-random',
                     'rewiring-edges', 'mode-collapse', 'mode-dropping',
                     'computation-eff-qty', 'computation-eff-size',
                     'computation-eff-edges', 'randomize-nodes',
                     'randomize-edges'],
            help='The permutation (experiment) to run')

        self.parser.add_argument(
            '--step_size', default=0.01, type=float,
            help='Many experiments have a "step size", e.g. in mixing random\
                graphs, the step size is the percentage (fraction) of random\
                graphs added at each time step.')

        self.parser.add_argument(
            '--results_directory', type=str, default='testing',
            help='Results are saved in experiment_results/{results_directory}')

        self.parser.add_argument(
            '--save_perturb', default=False, type=bool,
            help='If true it will save the graphs from the result of the perturbation.'
        )

        self.parser.add_argument(
            '--load_perturb', default=False, type=bool,
            help='If true it will load the previously perturbed data instead of generating it (if available).'
        )

        self.parser.add_argument(
            '--is_parallel', default=False, type=bool,
            help="For degree, clustering, orbits, and spectral MMD metrics. Or node feature extractor in GIN.\
            Whether to compute graph statistics in parallel or not.")

        self.parser.add_argument(
            '--max_workers', default=4, type=int,
            help="If is_parallel is true, this sets the maximum number of\
                    workers.")

        self.parser.add_argument(
            '--use_predefined_hyperparams', default=False, type=bool,
            help="If it is true, it uses hyperparameters predefined from before for each dataset.\
                 This ignores the hyperparameters in the config/args"
        )

        self.parser.add_argument(
            '--split', default=False, type=bool,
            help="Split the data between the train and test?")

        self.parser.add_argument(
            '--split_ratio', default=0.5, type=float,
            help="Ratio to split dataset between train and test data.")

        # gin_parser = subparsers.add_parser('gnn')
        self.parser.add_argument(
            '--feature_extractor', default='graphcl',
            choices=['graphcl', 'gin-random', 'infograph'],
            help='The GNN to use')

        self.parser.add_argument(
            '--num_layers', default=3, type=int,
            help='The number of prop. rounds in the GNN')

        self.parser.add_argument(
            '--hidden_dim', default=32, type=int,
            help='The node embedding dimensionality. Final graph embed size \
            is hidden_dim * (num_layers - 1)')

        self.parser.add_argument(
            '--epochs', default=100, type=int,
            help='number of epochs for training')

        self.parser.add_argument(
            '--init', default='orthogonal', type=str,
            choices=['default', 'orthogonal'],
            help="The weight init. method for the GNN. Default is PyTorchs\
            default init.")

        self.parser.add_argument(
            '--retrain', default=True, type=bool,
            help='Yes: retrain the gin network/ No: use a pretrained one if available')

        self.parser.add_argument(
            '--model_name', default='temp', type=str,
            help='name of the model to save')

        self.parser.add_argument(
            '--limit_lip', default=True, type=bool,
            help='Should model limit the lipchitzness factor of the layers?')

        self.parser.add_argument(
            '--lip_factor', default=1.0, type=float,
            help='lipchitzness factor of the mlp layers in GConv'
        )

        self.parser.add_argument(
            '--const_feats', default=True, type=bool,
            help='Should dataset add constant features?')

        self.parser.add_argument(
            '--deg_feats', action='store_true',
            # '--deg_feats', default=False, type=bool,
            help='Should dataset add target normalized degree features?') # default changed to featureless

        self.parser.add_argument(
            '--clus_feats', action='store_true',
            help='Should dataset add clustering features?')

        self.parser.add_argument(
            '--orbit_feats', action='store_true',
            help='Should dataset add orbit features?')

        self.parser.add_argument('--verbose', action='store_false')

        # Parser for the non-GNN-based metrics
        mmd_parser = subparsers.add_parser('mmd-structure')

        mmd_parser.add_argument(
            '--feature_extractor', default='mmd-structure',
            choices=['mmd-structure'])

        mmd_parser.add_argument(
            '--kernel', default='gaussian_emd',
            choices=['gaussian_emd', 'gaussian_rbf'],
            help="The kernel to use for the degree, clustering, orbits, and\
                   spectral MMD metrics. Gaussian EMD is the RBF kernel with the L2\
                   norm replaced by EMD.")

        mmd_parser.add_argument(
            '--is_parallel', action='store_true',
            help="For degree, clustering, orbits, and spectral MMD metrics.\
                   Whether to compute graph statistics in parallel or not.")

        mmd_parser.add_argument(
            '--max_workers', default=8, type=int,
            help="If is_parallel is true, this sets the maximum number of\
                   workers.")

        mmd_parser.add_argument(
            '--statistic', default='degree',
            choices=['degree', 'clustering', 'orbits', 'spectral',
                     'WL', 'nspdk'],
            help="The metric to use")

        mmd_parser.add_argument(
            '--sigma', default='single',
            choices=['single', 'range'],
            help="For degree, clustering, orbits, and spectral MMD metrics.\
                   Selects whether to use a single sigma (as in GraphRNN and GRAN),\
                   or to use the adaptive sigma we propose.")

        graphmae_parser = subparsers.add_parser('graphmae')
        graphmae_parser.add_argument('--feature_extractor', default='graphmae', choices=['graphmae'])
        # graphmae_parser.add_argument("--seeds", type=int, nargs="+", default=[0])
        # graphmae_parser.add_argument("--dataset", type=str, default="lobster")
        graphmae_parser.add_argument("--device", type=int, default=0)
        graphmae_parser.add_argument("--max_epoch", type=int, default=100,
                            help="number of training epochs")
        graphmae_parser.add_argument("--warmup_steps", type=int, default=-1)

        graphmae_parser.add_argument("--num_heads", type=int, default=4,
                            help="number of hidden attention heads")
        graphmae_parser.add_argument("--num_out_heads", type=int, default=1,
                            help="number of output attention heads")
        graphmae_parser.add_argument("--num_layers", type=int, default=3,
                            help="number of hidden layers")
        graphmae_parser.add_argument("--num_hidden", type=int, default=32,
                            help="number of hidden units")
        graphmae_parser.add_argument("--residual", action="store_true", default=False,
                            help="use residual connection")
        graphmae_parser.add_argument("--in_drop", type=float, default=.1,
                            help="input feature dropout")
        # graphmae_parser.add_argument("--attn_drop", type=float, default=.1,
        #                     help="attention dropout")
        graphmae_parser.add_argument("--norm", type=str, default='batchnorm')
        graphmae_parser.add_argument("--lr", type=float, default=0.001,
                            help="learning rate")
        graphmae_parser.add_argument("--weight_decay", type=float, default=0.,
                            help="weight decay")
        # graphmae_parser.add_argument("--negative_slope", type=float, default=0.2,
        #                     help="the negative slope of leaky relu for GAT")
        graphmae_parser.add_argument("--activation", type=str, default="prelu")
        graphmae_parser.add_argument("--mask_rate", type=float, default=0.2)
        graphmae_parser.add_argument("--drop_edge_rate", type=float, default=0.2, help='if to drop edge (edge_p>0), then the percentage to drop')
        graphmae_parser.add_argument("--replace_rate", type=float, default=0.0)

        graphmae_parser.add_argument("--encoder", type=str, default="gin")
        graphmae_parser.add_argument("--decoder", type=str, default="mlp") #$
        # graphmae_parser.add_argument("--decoder", type=str, default="gin")  # $
        graphmae_parser.add_argument("--loss_fn", type=str, default="sce")
        graphmae_parser.add_argument("--alpha_l", type=float, default=1, help="`pow`coefficient for `sce` loss")
        graphmae_parser.add_argument("--optimizer", type=str, default="adam")
        graphmae_parser.add_argument("--load_model", action="store_true")
        graphmae_parser.add_argument("--save_model", action="store_false")
        graphmae_parser.add_argument("--use_cfg", action="store_false")
        graphmae_parser.add_argument("--logging", action="store_true")
        graphmae_parser.add_argument("--scheduler", action="store_true", default=False)
        graphmae_parser.add_argument("--concat_hidden", action="store_true") # not work well for proteins 6/7/2024

        # for graph classification
        graphmae_parser.add_argument("--pooling", type=str, default="sum")
        graphmae_parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
        graphmae_parser.add_argument("--batch_size", type=int, default=32)
        graphmae_parser.add_argument('--node_label', action='store_false')
        graphmae_parser.add_argument('--use_edge_attr', action='store_false')
        graphmae_parser.add_argument("--num_layers_decoder", type=int, default=2,
                            help="number of hidden layers for decoder")
        graphmae_parser.add_argument("--num_layers_edge_decoder", type=int, default=2,
                            help="number of hidden layers")
        graphmae_parser.add_argument('--beta', type=float, default=0.5)
        graphmae_parser.add_argument('--edge_p', type=float, default=0.5, help="the probability to mask edges instead of masking nodes")
        # graphmae_parser.add_argument('--like_contrast', action='store_false')
        graphmae_parser.add_argument("--z_dim", type=int, default=32,
                            help="number of z dim")
        graphmae_parser.add_argument("--z_ratio", type=int, default=2,
                            help="number of z dim")

        maskgae_parser = subparsers.add_parser('maskgae')
        maskgae_parser.add_argument('--feature_extractor', default='maskgae', choices=['maskgae'])
        # maskgae_parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
        maskgae_parser.add_argument("--mask", nargs="?", default="Path",
                            help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
        # maskgae_parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

        maskgae_parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
        maskgae_parser.add_argument("--encoder_activation", nargs="?", default="elu",
                            help="Activation function for GNN encoder, (default: elu)")
        maskgae_parser.add_argument('--encoder_channels', type=int, default=128,
                            help='Channels of GNN encoder layers. (default: 128)')
        maskgae_parser.add_argument('--hidden_channels', type=int, default=64,
                            help='Channels of hidden representation. (default: 64)')
        maskgae_parser.add_argument('--decoder_channels', type=int, default=32,
                            help='Channels of decoder layers. (default: 128)')
        maskgae_parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
        maskgae_parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
        maskgae_parser.add_argument('--encoder_dropout', type=float, default=0.2,
                            help='Dropout probability of encoder. (default: 0.8)')
        maskgae_parser.add_argument('--decoder_dropout', type=float, default=0.2,
                            help='Dropout probability of decoder. (default: 0.2)')
        maskgae_parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

        maskgae_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training. (default: 0.01)')
        maskgae_parser.add_argument('--weight_decay', type=float, default=5e-5,
                            help='weight_decay for link prediction training. (default: 5e-5)')
        maskgae_parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
        maskgae_parser.add_argument('--batch_size', type=int, default=32,
                            help='Number of batch size for link prediction training. (default: 2**16)')

        maskgae_parser.add_argument("--start", nargs="?", default="node",
                            help="Which Type to sample starting nodes for random walks, (default: node)")
        maskgae_parser.add_argument('--p', type=float, default=0.3, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

        maskgae_parser.add_argument('--bn', action='store_true',
                            help='Whether to use batch normalization for GNN encoder. (default: False)')
        maskgae_parser.add_argument('--l2_normalize', action='store_true',
                            help='Whether to use l2 normalize output embedding. (default: False)')
        maskgae_parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                            help='weight_decay for node classification training. (default: 1e-3)')

        maskgae_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs. (default: 500)')
        maskgae_parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
        maskgae_parser.add_argument('--eval_period', type=int, default=1, help='(default: 30)')
        maskgae_parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
        maskgae_parser.add_argument("--save_path", nargs="?", default="model_nodeclas",
                            help="save path for model. (default: model_nodeclas)")
        maskgae_parser.add_argument('--debug', action='store_true',
                            help='Whether to log information in each epoch. (default: False)')

        maskgae_parser.add_argument("--device", type=int, default=0)
        maskgae_parser.add_argument('--node_label', action='store_false')
        maskgae_parser.add_argument('--use_edge_attr', action='store_false')
        maskgae_parser.add_argument('--edge_batch_size', type=int, default=2 ** 8)

    def parse(self):
        """Parse the given command line arguments.
        Parses the command line arguments and overwrites
        some values to ensure compatibility.
        Returns
        -------
        Argparse dict: The parsed CL arguments
        """
        args = self.parser.parse_args()
        args.results_directory = '' if args.results_directory is None \
            else args.results_directory + '/'

        # args.use_degree_features = True

        if args.dataset == 'zinc':
            args.input_dim = 28  # The number of node features in zinc
            args.edge_feat_dim = 4  # Num edge feats. in zinc
        else:
            args.input_dim = 1  # We use node degree as an int. as node feats.
            args.edge_feat_dim = None  # No edge features for non-zinc datasets

        args.results_directory += \
            f'{args.feature_extractor}/{args.permutation_type}/{args.dataset}'
        args.graph_embed_size = (args.num_layers - 1) * args.hidden_dim

        if args.use_predefined_hyperparams:
            if args.dataset == 'lobster':
                args.hidden_dim = 16
                args.num_layers = 2
                args.epochs = 40
            elif args.dataset == 'grid':
                args.hidden_dim = 16
                args.num_layers = 3
                args.epochs = 20
            else:
                args.hidden_dim = 32
                args.num_layers = 3
                args.epochs = 100

        if args.model_name == 'temp':
            args.model_name = f'{args.feature_extractor}_{args.dataset}_{args.num_layers}_{args.hidden_dim}_def_feat_{args.deg_feats}_cluster_feat_{args.clus_feats}' \
                              f'_{args.orbit_feats}_{args.epochs}_{args.limit_lip}_{args.lip_factor}_{args.split_ratio}'

        args.permute_data_drc = f'data/perturb_data/{args.permutation_type}/{args.dataset}/{args.seed}'

        # from graphmae.utils import load_best_configs
        # if args.feature_extractor == 'graphmae':
        #     args = load_best_configs(args, './configs.yml')

        if 'node_label' not in args:
            args.node_label = True

        return args
