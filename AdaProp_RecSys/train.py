"""
Training script for AdaProp on RecSys (KUCNet-style) datasets.

Usage
-----
    python train.py --data_path data/last-fm/ --topk 50 --layers 3 --gpu 0
    python train.py --data_path data/amazon-book/ --topk 120 --layers 3 --gpu 0
    python train.py --data_path data/alibaba-fashion/ --topk 70 --layers 5 --gpu 0

The hyper-parameter presets below mirror the KUCNet defaults but add
AdaProp-specific knobs (``--topk``, ``--tau``, ``--layers``).
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from load_data import DataLoader
from base_model import BaseModel
from utils import checkPath, seed_everything

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from exp_logger import ExpLogger

# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description='AdaProp for Recommendation')
parser.add_argument('--data_path', type=str, default='data/last-fm/',
                    help='Path to dataset folder')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--topk', type=int, default=50,
                    help='Node-sampling budget per layer (AdaProp top-k)')
parser.add_argument('--layers', type=int, default=3,
                    help='Number of GNN propagation layers')
parser.add_argument('--tau', type=float, default=1.0,
                    help='Gumbel-softmax temperature for adaptive sampling')
parser.add_argument('--ppr_topk', type=int, default=0,
                    help='PPR edge-pruning top-k per node (0=disabled)')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--scheduler', type=str, default='exp')
parser.add_argument('--weight', type=str, default=None,
                    help='Path to a saved checkpoint to resume from')
args = parser.parse_args()


class Options:
    """Lightweight namespace that is populated at runtime."""
    pass


if __name__ == '__main__':
    seed_everything(args.seed)
    torch.set_num_threads(8)
    print(f'# seed: {args.seed}')

    dataset = args.data_path.rstrip('/').split('/')[-1]
    torch.cuda.set_device(args.gpu)
    print(f'==> gpu: {args.gpu}')

    # ---- data ----
    loader = DataLoader(args.data_path)

    opts = Options()
    opts.n_ent   = loader.n_ent
    opts.n_rel   = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes
    opts.tau     = args.tau
    opts.scheduler = args.scheduler

    # ---------------------------------------------------------------
    # Per-dataset hyper-parameter presets
    # (base values from KUCNet, plus AdaProp-specific topk / tau)
    # ---------------------------------------------------------------
    if dataset == 'new_alibaba-fashion':
        opts.lr         = 0.00005
        opts.decay_rate = 0.999
        opts.lamb       = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'alibaba-fashion':
        opts.lr         = 10 ** -6.5
        opts.decay_rate = 0.998
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.2
        opts.act        = 'relu'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'last-fm':
        opts.lr         = 0.0004
        opts.decay_rate = 0.994
        opts.lamb       = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'new_last-fm':
        opts.lr         = 0.0004
        opts.decay_rate = 0.994
        opts.lamb       = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'new_amazon-book':
        opts.lr         = 0.0005
        opts.decay_rate = 0.994
        opts.lamb       = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'amazon-book':
        opts.lr         = 0.0012
        opts.decay_rate = 0.994
        opts.lamb       = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'Dis_5fold_item':
        opts.lr         = 0.0005
        opts.decay_rate = 0.994
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    elif dataset == 'Dis_5fold_user':
        opts.lr         = 0.001
        opts.decay_rate = 0.994
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    else:
        # sensible fallback for unknown datasets
        opts.lr         = 0.0002
        opts.decay_rate = 0.9938
        opts.lamb       = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.layers
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 5
        opts.n_tbatch   = 5

    # AdaProp-specific: per-layer topk budget
    opts.n_node_topk = [args.topk] * opts.n_layer
    opts.n_edge_topk = -1          # no edge sampling by default
    opts.ppr_topk    = args.ppr_topk

    # ---- output dirs ----
    checkPath('./results/')
    save_model_dir = os.path.join(loader.task_dir, 'saveModel')
    checkPath(save_model_dir)

    opts.perf_file = os.path.join('results', f'{dataset}_perf.txt')
    print(f'==> perf_file: {opts.perf_file}')

    config_str = (f'topk={args.topk}, tau={opts.tau}, lr={opts.lr:.6f}, '
                  f'decay={opts.decay_rate:.4f}, lamb={opts.lamb:.6f}, '
                  f'dim={opts.hidden_dim}, attn={opts.attn_dim}, '
                  f'layers={opts.n_layer}, batch={opts.n_batch}, '
                  f'drop={opts.dropout:.4f}, act={opts.act}\n')
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    # ---- model ----
    model = BaseModel(opts, loader)
    n_params_total = sum(p.numel() for p in model.model.parameters())
    n_params_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f'# model params: {n_params_total:,} (trainable: {n_params_trainable:,})')
    with open(opts.perf_file, 'a+') as f:
        f.write(f'# model_params_total: {n_params_total}\n')
        f.write(f'# model_params_trainable: {n_params_trainable}\n')

    # ---- PPR edge pruning (compute / load cached) ----
    if opts.ppr_topk > 0:
        from ppr import get_ppr_cached
        ppr_indices, ppr_values = get_ppr_cached(loader, topk=opts.ppr_topk)
        model.model.set_ppr(ppr_indices, ppr_values)
        print(f'PPR edge pruning enabled: top-{opts.ppr_topk} per node')

    if args.weight is not None:
        model.loadModel(args.weight)
        model._update()
        model.model.updateTopkNums(opts.n_node_topk)

    # ---- experiment logger ----
    log_config = {
        'topk': args.topk, 'tau': opts.tau, 'lr': opts.lr,
        'decay_rate': opts.decay_rate, 'lamb': opts.lamb,
        'hidden_dim': opts.hidden_dim, 'attn_dim': opts.attn_dim,
        'n_layer': opts.n_layer, 'dropout': opts.dropout, 'act': opts.act,
        'n_batch': opts.n_batch, 'ppr_topk': opts.ppr_topk,
        'seed': args.seed, 'gpu': args.gpu, 'epoch': args.epoch,
    }
    exp_log = ExpLogger(
        model_name='adaprop', dataset=dataset, config=log_config,
        log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', 'adaprop'),
    )

    # ---- training loop ----
    best_recall = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    best_str = ''
    for epoch in tqdm(range(args.epoch), desc='[AdaProp] epochs', unit='epoch'):
        recall, ndcg, out_str, train_time, eval_time = model.train_batch()
        exp_log.log_epoch(epoch, train_time, eval_time, recall, ndcg)

        with open(opts.perf_file, 'a+') as f:
            f.write(f'epoch {epoch}  {out_str}')

        if recall > best_recall:
            best_recall = recall
            best_ndcg = ndcg
            best_epoch = epoch
            best_str = out_str
            tqdm.write(f'  *** new best @ epoch {epoch}: {best_str.strip()}')
            model.saveModel(f'recall_{recall:.4f}', delete_last=True)

    exp_log.finish(best_epoch, best_recall, best_ndcg)

    with open(opts.perf_file, 'a+') as f:
        f.write(f'best:\n{best_str}')
    print(f'\n==> Best result: {best_str}')
