from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    num_layers_decoder = args.num_layers_decoder
    num_layers_edge_decoder = args.num_layers_edge_decoder
    beta = args.beta
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    drop_edge_rate = args.drop_edge_rate
    replace_rate = args.replace_rate
    edge_p = args.edge_p
    # z_dim = args.z_dim
    z_ratio = args.z_ratio

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        edge_dim = args.edge_dim if 'edge_dim' in args else None,
        num_layers_decoder = num_layers_decoder,
        num_layers_edge_decoder = num_layers_edge_decoder,
        beta=beta,
        edge_p=edge_p,
        # z_dim=z_dim,
        z_ratio=z_ratio
    )
    return model
