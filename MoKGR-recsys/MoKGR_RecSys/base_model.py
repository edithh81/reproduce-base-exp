"""
Training / evaluation harness for MoKGR on RecSys datasets.

Training uses BPR loss + MoE regularization losses.
Evaluation uses Recall@K and NDCG@K (K=20 by default).
"""

import torch
import numpy as np
import time
import heapq

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import MoKGR_RecSys
from utils import cal_bpr_loss, ndcg_k
import logging
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, args, loader):
        self.current_epoch = 0

        self.logger = logging.getLogger('BaseModelLogger')
        self.logger.setLevel(logging.INFO)
        self.model = MoKGR_RecSys(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_users = loader.n_users
        self.n_items = loader.n_items
        self.n_nodes = loader.n_nodes
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.known_user_set = loader.known_user_set
        self.test_user_set = loader.test_user_set

        if not self.logger.handlers:
            fh = logging.FileHandler(args.log_file)
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        self.logger.info('Model initialization is complete.')

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0
        self.lambda_importance = args.lambda_importance
        self.lambda_load = args.lambda_load
        self.lambda_importance_pruning = args.lambda_importance_pruning

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train_batch(self):
        self.current_epoch += 1
        epoch_loss = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)

        torch.cuda.reset_peak_memory_stats()
        t_time = time.time()
        self.model.train()
        for i in tqdm(range(n_batch), desc="Training Epoch Progress", unit="batch"):
            start = i * batch_size
            end = min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)

            subs, rels, pos, neg = self.loader.get_batch(batch_idx, data='train')

            self.model.zero_grad()
            scores, G_full, Q, L_importance_pruning = self.model(subs, rels, mode='train')

            # BPR loss
            loss = cal_bpr_loss(self.n_users, pos, neg, scores)

            # MoE regularization losses (architectural, not task-specific)
            L_importance = self.model.moe_for_hops.compute_importance_loss(G_full)
            L_load = self.model.moe_for_hops.compute_load_loss(Q, G_full)

            total_loss = (loss
                          + self.lambda_importance * L_importance
                          + self.lambda_load * L_load
                          + self.lambda_importance_pruning * L_importance_pruning)

            total_loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

            if i % 500 == 0:
                print(f'  batch {i}  loss={loss.item():.4f}')

        self.scheduler.step()
        train_time = time.time() - t_time
        train_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.t_time += train_time
        print(f'[TRAIN] time: {train_time:.2f}s | peak CUDA mem: {train_peak_mem:.2f} MB')

        self.loader.shuffle_train()
        print(f'epoch_loss = {epoch_loss:.4f}')

        # run evaluation after each epoch
        recall, ndcg, out_str, eval_time = self.test_batch()
        self.logger.info(out_str)
        return recall, ndcg, out_str, train_time, eval_time

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def test_one_user(self, u, score, K=20):
        try:
            training_items = self.known_user_set[u]
        except KeyError:
            training_items = []
        user_pos_test = self.test_user_set[u]

        all_items = set(range(self.n_users, self.n_users + self.n_items))
        test_items = list(all_items - set(training_items))

        item_score = {}
        for it in test_items:
            item_score[it] = score[it - self.n_users]

        top_items = heapq.nlargest(K, item_score, key=item_score.get)

        r = [1 if it in user_pos_test else 0 for it in top_items]
        ndcg = ndcg_k(r, K, len(user_pos_test))
        recall = np.sum(r) / max(len(user_pos_test), 1)
        return recall, ndcg

    def test_batch(self, K=20):
        batch_size = self.n_tbatch
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        self.model.eval()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        recall_sum, ndcg_sum = 0.0, 0.0

        with torch.no_grad():
            for bid in tqdm(range(n_batch), desc='eval'):
                start = bid * batch_size
                end = min(n_data, (bid + 1) * batch_size)
                batch_idx = np.arange(start, end)

                subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
                scores = self.model(subs, rels, mode='test').data.cpu().numpy()

                for j in range(len(subs)):
                    u = subs[j]
                    one_r, one_n = self.test_one_user(u, scores[j], K=K)
                    recall_sum += one_r
                    ndcg_sum += one_n

                if bid % 500 == 0:
                    print(f'  eval batch {bid}  recall(batch)='
                          f'{recall_sum / max((bid + 1) * batch_size, 1):.4f}')

        recall = recall_sum / n_data
        ndcg = ndcg_sum / n_data
        i_time = time.time() - t0
        inf_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'[INFERENCE] time: {i_time:.2f}s | peak CUDA mem: {inf_peak_mem:.2f} MB')
        out_str = (f'[Epoch {self.current_epoch}] [TEST] Recall@{K}: {recall:.4f}  '
                   f'NDCG@{K}: {ndcg:.4f}  '
                   f'[TIME] train: {self.t_time:.2f}s  eval: {i_time:.2f}s  '
                   f'[MEM] inf_peak: {inf_peak_mem:.2f} MB\n')
        return recall, ndcg, out_str, i_time
