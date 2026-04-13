import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from load_data import DataLoader
from base_model import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from exp_logger import ExpLogger

parser = argparse.ArgumentParser(description="Parser for KUCNet")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=40)

args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    torch.cuda.set_device(args.gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users   
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    if dataset == 'new_alibaba-fashion':  
        opts.lr = 0.00005
        opts.decay_rate = 0.999
        opts.lamb = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 50
    elif dataset == 'alibaba-fashion'  :
        opts.lr = 10**-6.5
        opts.decay_rate = 0.998
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 70
    elif dataset == 'last-fm' :
        opts.lr = 0.0004
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 35
    elif dataset == 'new_last-fm' :
        opts.lr = 0.0004
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 50
    elif dataset == 'new_amazon-book':
        opts.lr = 0.0005
        opts.decay_rate = 0.994
        opts.lamb = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 170
    elif dataset == 'amazon-book' :
        opts.lr = 0.0012
        opts.decay_rate = 0.994
        opts.lamb = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 120
    elif dataset == 'Dis_5fold_item'   :
        opts.lr = 0.0005
        opts.decay_rate = 0.994
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 35
    elif dataset == 'Dis_5fold_user'   :
        opts.lr = 0.001
        opts.decay_rate = 0.994
        opts.lamb = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.01
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = 550
    else:
        opts.lr = 0.0002
        opts.decay_rate = 0.9938
        opts.lamb = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 5
        opts.n_tbatch = 5
        opts.K = args.K


    config_str = '%d,%.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.K,opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    # ---- experiment logger ----
    log_config = {
        'K': opts.K, 'lr': opts.lr, 'decay_rate': opts.decay_rate,
        'lamb': opts.lamb, 'hidden_dim': opts.hidden_dim, 'attn_dim': opts.attn_dim,
        'n_layer': opts.n_layer, 'dropout': opts.dropout, 'act': opts.act,
        'n_batch': opts.n_batch, 'seed': args.seed, 'gpu': args.gpu,
    }
    exp_log = ExpLogger(
        model_name='kucnet', dataset=dataset, config=log_config,
        log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', 'kucnet'),
    )

    n_epochs = args.epoch
    best_recall = 0
    best_epoch = 0
    best_ndcg = 0
    for epoch in tqdm(range(n_epochs), desc='[KUCNet] epochs', unit='epoch'):
        recall, ndcg, out_str, train_time, eval_time = model.train_batch()
        exp_log.log_epoch(epoch, train_time, eval_time, recall, ndcg)

        with open(opts.perf_file, 'a+') as f:
            f.write(str(epoch) + out_str)

        if recall > best_recall:
            best_recall = recall
            best_ndcg = ndcg
            best_epoch = epoch
            best_str = out_str
            tqdm.write(f'epoch {epoch}\t{best_str.strip()}')

    exp_log.finish(best_epoch, best_recall, best_ndcg)

    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n'+best_str)

    print(best_str)

