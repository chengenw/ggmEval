import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, MLP, GINEConv
from graphmae.utils import create_activation, NormLayer, create_norm
from termcolor import cprint


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 edge_dim=None
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if edge_dim is not None:
            cprint(f'edge_dim is {edge_dim}', 'red')

        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            if edge_dim is None:
                self.layers.append(GINConv(nn=apply_func, train_eps=False))
            else:
                self.layers.append(GINEConv(nn=apply_func, train_eps=False, edge_dim=edge_dim))
        else:
            # input projection (no residual)
            if edge_dim is None:
                self.layers.append(GINConv(
                    nn=ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                    train_eps=False)
                    )
            else:
                self.layers.append(GINEConv(
                    nn=ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                    train_eps=False, edge_dim=edge_dim)
                    )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                if edge_dim is None:
                    self.layers.append(GINConv(
                        nn=ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                        train_eps=False)
                    )
                else:
                    self.layers.append(GINEConv(
                        nn=ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                        train_eps=False, edge_dim=edge_dim)
                    )
            # output projection
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            if edge_dim is None:
                self.layers.append(GINConv(nn=apply_func, train_eps=False))
            else:
                self.layers.append(GINEConv(nn=apply_func, train_eps=False, edge_dim=edge_dim))

        self.head = nn.Identity()

    def forward(self, inputs, edge_index, return_hidden=False, edge_attr=None):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](h, edge_index, edge_attr)
            # if edge_attr:
            #     h = self.layers[l](h, edge_index, edge_attr)
            # else:
            #     h = self.layers[l](h, edge_index)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

        self.in_features = self.mlp.in_features #$

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.in_features = input_dim  #$

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)