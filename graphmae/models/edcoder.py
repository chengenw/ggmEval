from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gat import GAT
from .gin import GIN
from .loss_func import sce_loss
from graphmae.utils import create_norm, dropout_edge
# from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MLP
from torch_geometric.utils import batched_negative_sampling

from termcolor import cprint

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True, edge_dim=None) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            edge_dim=edge_dim
        )
    elif m_type == "mlp":
        # * just for decoder 
        # mod = nn.Sequential(
        #     nn.Linear(in_dim, num_hidden),
        #     nn.PReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(num_hidden, out_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(num_hidden, out_dim),
        #     # nn.ReLU(),
        # )
        mod = MLP(in_channels=in_dim, hidden_channels=num_hidden, out_channels=out_dim, num_layers=2)
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            edge_dim = None,
            num_layers_decoder = 2,
            num_layers_edge_decoder = 2,
            beta = 1,
            edge_p=0.5,
            # z_dim=None
            z_ratio=1
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self.beta = beta
        self.edge_p = edge_p


        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        # dec_in_dim = z_dim
        dec_in_dim = enc_num_hidden // z_ratio
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=dec_in_dim,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            edge_dim=edge_dim
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=num_layers_decoder,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.edge_decoder = setup_module(
            # m_type=decoder_type,
            m_type='mlp',
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=dec_num_hidden,
            num_layers=num_layers_edge_decoder,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim)) #$ [MASK]
        if self._concat_hidden:
            self.encoder_to_decoder = nn.Linear(enc_num_hidden * (num_layers - 1) + dec_in_dim, dec_in_dim, bias=False)
        else:
            # self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            self.encoder_to_decoder = nn.Identity()

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0: #$ equivalent to reducing mask rate by self._replace_rate?
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward_(self, x, edge_index, edge_attr=None, batch=None):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(x, edge_index, edge_attr=edge_attr, batch=batch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        prob = torch.rand(1)
        if prob > self.edge_p:
            cur_mask = 'node'
            # cprint(f'masked attributes prediction', 'yellow')
            loss = self.mask_attr_prediction(x, edge_index, edge_attr=edge_attr, batch=batch)
        else:
            cur_mask = 'edge'
            # cprint(f'masked edges prediction', 'yellow')
            loss = self.mask_edge_prediction(x, edge_index, edge_attr=edge_attr, batch=batch)
        loss_item = {f"loss_{cur_mask}": loss.item()}
        return loss, loss_item

    def mask_attr_prediction_(self, x, edge_index, edge_attr=None, batch=None):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        if self._drop_edge_rate > 0:
            # use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate, force_undirected=False)
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate, force_undirected=True)
            use_edge_attr = edge_attr[masked_edges] if edge_attr is not None else None
            fill_v = torch.zeros(edge_attr.shape[1]) if edge_attr is not None else None
            use_edge_index, use_edge_attr = add_self_loops(use_edge_index, use_edge_attr, fill_v)

        else:
            use_edge_index = edge_index
            use_edge_attr = edge_attr

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=True, edge_attr=use_edge_attr)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon, h_list = self.decoder(rep, use_edge_index, return_hidden=True)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        recon_edge, h_list_edge = self.edge_decoder(rep, use_edge_index, return_hidden=True)
        z = h_list_edge[-1]
        loss_edge = self.edge_loss(z, edge_index, ~masked_edges, batch)
        loss = (1 - self.beta) * loss + self.beta * loss_edge
        # loss += self.beta * loss_edge

        return loss

    def mask_attr_prediction(self, x, edge_index, edge_attr=None, batch=None):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        use_edge_index = edge_index
        use_edge_attr = edge_attr

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=True, edge_attr=use_edge_attr)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon, h_list = self.decoder(rep, use_edge_index, return_hidden=True)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        return loss

    def mask_edge_prediction(self, x, edge_index, edge_attr=None, batch=None):
        # use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate, force_undirected=False)
        use_edge_index, keep_mask, drop_mask = dropout_edge(edge_index, self._drop_edge_rate, force_undirected=True)
        # if edge_attr is not None:
        #     raise NotImplementedError
        use_edge_attr = torch.cat((edge_attr[keep_mask], edge_attr[keep_mask]), dim=0) if edge_attr is not None else None
        fill_v = torch.zeros(edge_attr.shape[1]) if edge_attr is not None else None
        use_edge_index, use_edge_attr = add_self_loops(use_edge_index, use_edge_attr, fill_v)

        enc_rep, all_hidden = self.encoder(x, use_edge_index, return_hidden=True, edge_attr=use_edge_attr)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        # _, h_list_edge = self.edge_decoder(rep, use_edge_index, return_hidden=True)
        # z = h_list_edge[-1]
        z = self.edge_decoder(rep)

        drop_edges = torch.cat([edge_index[:, drop_mask], edge_index[:, drop_mask].flip(0)], dim=1)
        loss = self.edge_loss(z, edge_index, drop_edges, batch)

        return loss

    def edge_loss(self, z, edge_index, drop_edges, batch=None):
        EPS = 1e-15
        def inner_product(z, edge_index, sigmoid=True):
            value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            return torch.sigmoid(value) if sigmoid else value

        pos_loss = -torch.log(
            inner_product(z, drop_edges, sigmoid=True) + EPS).mean()

        # neg_edge_index = batched_negative_sampling(edge_index, batch)
        # neg_edge_index = batched_negative_sampling(edge_index, batch, force_undirected=True, num_neg_samples=edge_index.size(1))
        neg_edge_index = batched_negative_sampling(edge_index, batch, force_undirected=True)
        neg_loss = -torch.log(1 -
                              inner_product(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def embed(self, x, edge_index, edge_attr=None):
        rep, all_hidden = self.encoder(x, edge_index, edge_attr=edge_attr, return_hidden=True)
        if self._concat_hidden:
            rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(rep)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def get_graph_embed(self, x, edge_index, batch, edge_attr=None):
        # cprint(f'edcoder.py::get_graph_embed self.eval()!', 'red') #$!
        self.eval()
        out = self.embed(x, edge_index, edge_attr=edge_attr)
        if self.pooler == "mean":
            out = global_mean_pool(out, batch)
        elif self.pooler == "max":
            out = global_max_pool(out, batch)
        elif self.pooler == "sum":
            out = global_add_pool(out, batch)
        else:
            raise NotImplementedError

        return out
