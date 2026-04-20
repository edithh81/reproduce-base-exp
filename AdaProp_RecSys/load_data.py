"""
DataLoader for AdaProp on RecSys datasets.
Mirrors KUCNet's data loading (user-item CF + KG), but exposes
``get_neighbors(nodes, batchsize, mode)`` with the batchsize argument
that AdaProp's adaptive-sampling GNNLayer expects.
"""

import os
import random
import torch
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


class DataLoader:
    def __init__(self, task_dir):
        self.task_dir = task_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._train_gpu_csr = None
        self._test_gpu_csr = None

        # ---- read user-item interactions ----
        if any(tag in task_dir for tag in
               ['Dis_5fold_user', 'Dis_5fold_item',
                'new_last-fm', 'new_amazon-book', 'new_alibaba-fashion']):
            self.all_cf  = self.read_cf(os.path.join(task_dir, 'train_1.txt'))
            self.test_cf = self.read_cf(os.path.join(task_dir, 'test_1.txt'))
        else:
            self.all_cf  = self.read_cf(os.path.join(task_dir, 'train.txt'))
            self.test_cf = self.read_cf(os.path.join(task_dir, 'test.txt'))

        self.n_users = max(max(self.all_cf[:, 0]), max(self.test_cf[:, 0])) + 1
        self.n_items = max(max(self.all_cf[:, 1]), max(self.test_cf[:, 1])) + 1
        self.known_user_set = self.cf_to_set(self.all_cf)
        self.test_user_set  = self.cf_to_set(self.test_cf)

        n_all = self.all_cf.shape[0]
        rand_idx = np.random.permutation(n_all)
        self.all_cf = self.all_cf[rand_idx]

        # ---- read KG triples ----
        self.triple = self.read_triples('kg.txt')
        self.arraytriple = np.asarray(self.triple)
        self.n_ent   = max(max(self.arraytriple[:, 0]), max(self.arraytriple[:, 2])) + 1
        self.n_nodes = self.n_ent + self.n_users   # user-ids live in [0, n_users)
        self.n_rel   = max(self.arraytriple[:, 1]) + 1

        # ---- split facts / train ----
        if any(tag in task_dir for tag in
               ['Dis_5fold_item', 'new_last-fm', 'new_amazon-book', 'new_alibaba-fashion']):
            self.item_set = self.cf_to_item_set(self.all_cf)
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
        else:
            self.facts_cf = self.all_cf[0:n_all * 6 // 7]
            self.train_cf = self.all_cf[n_all * 6 // 7:]

        self.fact_triple  = self.cf_to_triple(self.facts_cf)
        self.train_triple = self.cf_to_triple(self.train_cf)
        self.test_triple  = self.cf_to_triple(self.test_cf)

        # add inverse KG edges
        self.d_triple = self.double_triple(self.triple)

        # build full-graph triples (KG + user-item)
        self.fact_data, self.known_data = self.interact_triple(self.d_triple)

        # optional user-KG
        if any(tag in task_dir for tag in ['Dis_5fold_user', 'Dis_5fold_item']):
            self.readukg = 1
            self.ukg = self.read_user_kg()
            self.n_rel += 1
            self.fact_data  += self.ukg
            self.known_data += self.ukg
            print('loaded user-KG triples')
        else:
            self.readukg = 0

        self.load_graph(self.fact_data)
        self.load_test_graph(self.known_data)

        self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
        self.test_q, self.test_a = self.load_query(self.test_triple)

        self.n_train = len(self.train_q)
        self.n_test  = len(self.test_q)

        self._build_eval_item_lists()

        print(f'n_facts: {len(self.facts_cf)}  n_test_cf: {len(self.test_cf)}  '
              f'n_train: {self.n_train}  n_test: {self.n_test}')
        print(f'users: {self.n_users}  items: {self.n_items}  '
              f'other entities: {self.n_ent - self.n_items}')

    # ------------------------------------------------------------------
    # reading helpers
    # ------------------------------------------------------------------
    def read_cf(self, file_name):
        inter_mat = []
        with open(file_name, 'r') as f:
            for line in f:
                inters = [int(x) for x in line.strip().split()]
                u_id, pos_ids = inters[0], list(set(inters[1:]))
                for i_id in pos_ids:
                    inter_mat.append([u_id, i_id])
        return np.array(inter_mat)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                triples.append([int(h), int(r), int(t)])
        return triples

    def double_triple(self, triples):
        new_triples = []
        for h, r, t in triples:
            new_triples.append([t, r + self.n_rel, h])
        return triples + new_triples

    def interact_triple(self, triples):
        """Build full-graph triples with user-item interactions.

        Relation mapping:
            0  : interact  (user -> item)
            1  : inv-interact (item -> user)
            2+ : KG relations shifted by 2  (+ inverse at n_rel+2)
        Entity mapping: KG entity e becomes e + n_users in the graph.
        """
        copy_tri = []
        for h, r, t in triples:
            copy_tri.append([h + self.n_users, r + 2, t + self.n_users])

        fact_user_triple = []
        for u, _, i in self.fact_triple:
            fact_user_triple.append([u, 0, i])
            fact_user_triple.append([i, 1, u])

        train_user_triple = []
        for u, _, i in self.train_triple:
            train_user_triple.append([u, 0, i])
            train_user_triple.append([i, 1, u])

        return (copy_tri + fact_user_triple,
                copy_tri + fact_user_triple + train_user_triple)

    def cf_to_triple(self, cf):
        triples = []
        for u, i in cf.tolist():
            if u >= self.n_users:
                continue
            triples.append([u, 0, i + self.n_users])
        return triples

    def cf_to_set(self, cf):
        user_set = defaultdict(list)
        for u, i in cf.tolist():
            if u >= self.n_users:
                continue
            user_set[u].append(i + self.n_users)
        return user_set

    def cf_to_item_set(self, cf):
        item_set = defaultdict(list)
        for u, i in cf.tolist():
            if u >= self.n_users:
                continue
            item_set[i].append(u)
        return item_set

    def read_user_kg(self):
        ukg = []
        with open(os.path.join(self.task_dir, 'ukg.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = int(h), int(r), int(t)
                if h >= self.n_users or t >= self.n_users:
                    continue
                ukg.append([h, 2 * self.n_rel + 2, t])
                ukg.append([t, 2 * self.n_rel + 3, h])
        return ukg

    def generate_inductive_train(self, cf):
        fcf = cf.tolist()
        n_train = 0
        train_cf, ind_item = [], []
        while n_train < len(cf) / 8:
            item = random.randint(0, self.n_items - 1)
            if item in ind_item:
                continue
            for u in self.item_set[item]:
                train_cf.append([u, item])
                fcf.remove([u, item])
            ind_item.append(item)
            n_train += len(self.item_set[item])
        return np.array(fcf), np.array(train_cf)

    # ------------------------------------------------------------------
    # graph construction
    # ------------------------------------------------------------------
    def load_graph(self, triples):
        """Build adjacency for the fact graph (train-time).

        Self-loop relation id = 2 * n_rel + 2  (after the shift by 2 for
        interact/inv-interact relations).
        """
        self_rel_id = 2 * self.n_rel + 2
        idd = np.column_stack([
            np.arange(self.n_nodes),
            np.full(self.n_nodes, self_rel_id),
            np.arange(self.n_nodes),
        ])
        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix(
            (np.ones(self.n_fact), (np.arange(self.n_fact), self.KG[:, 0])),
            shape=(self.n_fact, self.n_nodes),
        )

    def load_test_graph(self, triples):
        self_rel_id = 2 * self.n_rel + 2
        idd = np.column_stack([
            np.arange(self.n_nodes),
            np.full(self.n_nodes, self_rel_id),
            np.arange(self.n_nodes),
        ])
        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix(
            (np.ones(self.tn_fact), (np.arange(self.tn_fact), self.tKG[:, 0])),
            shape=(self.tn_fact, self.n_nodes),
        )

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------
    def load_train_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        pos_items = defaultdict(list)
        neg_items = defaultdict(list)
        for h, r, t in triples:
            pos_items[(h, r)].append(t)
            while True:
                neg_item = np.random.randint(self.n_users, self.n_users + self.n_items)
                if neg_item not in self.known_user_set[h]:
                    break
            neg_items[(h, r)].append(neg_item)
        queries, answers, wrongs = [], [], []
        for key in pos_items:
            queries.append(key)
            answers.append(np.array(pos_items[key]))
            wrongs.append(np.array(neg_items[key]))
        return queries, answers, wrongs

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(list)
        for h, r, t in triples:
            trip_hr[(h, r)].append(t)
        queries, answers = [], []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    # ------------------------------------------------------------------
    # neighbor sampling  (matches AdaProp interface: includes batchsize)
    # ------------------------------------------------------------------
    def get_neighbors(self, nodes, batchsize, mode='train'):
        """Return one-hop neighbors of *nodes* packed into tensors.

        Parameters
        ----------
        nodes : ndarray  [N_prev, 2]  with (batch_idx, node_idx)
        batchsize : int  (needed by AdaProp's node-sampling logic)
        mode : 'train' | 'test'

        Returns
        -------
        tail_nodes : LongTensor [N_next, 2]
        sampled_edges : LongTensor [E, 6]
            (batch_idx, head, rela, tail, head_rel_idx, tail_rel_idx)
        old_nodes_new_idx : LongTensor [N_prev]
        """
        if mode == 'train':
            KG, M_sub = self.KG, self.M_sub
        else:
            KG, M_sub = self.tKG, self.tM_sub

        node_1hot = csr_matrix(
            (np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])),
            shape=(self.n_nodes, nodes.shape[0]),
        )
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate(
            [np.expand_dims(edges[1], 1), KG[edges[0]]], axis=1
        )
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(
            sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True
        )
        tail_nodes, tail_index = torch.unique(
            sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True
        )
        sampled_edges = torch.cat(
            [sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1
        )

        self_rel_id = 2 * self.n_rel + 2
        mask = sampled_edges[:, 2] == self_rel_id
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    # ------------------------------------------------------------------
    # vectorized eval tensors
    # ------------------------------------------------------------------
    def _build_eval_item_lists(self):
        """Precompute per-user item-local tensors for vectorized eval.

        ``known_user_set``/``test_user_set`` store items as ``i + n_users``;
        we subtract the offset so the tensors are item-local in [0, n_items).
        """
        self._known_items_local_per_user = [None] * self.n_users
        self._test_items_local_per_user = [None] * self.n_users
        empty = torch.empty(0, dtype=torch.long)
        for u in range(self.n_users):
            known = self.known_user_set.get(u, None)
            if known:
                self._known_items_local_per_user[u] = torch.tensor(
                    [i - self.n_users for i in known], dtype=torch.long
                )
            else:
                self._known_items_local_per_user[u] = empty
            test = self.test_user_set.get(u, None)
            if test:
                self._test_items_local_per_user[u] = torch.tensor(
                    [i - self.n_users for i in test], dtype=torch.long
                )
            else:
                self._test_items_local_per_user[u] = empty

    def get_eval_tensors(self, subs, device):
        """Return (known_mask [B, n_items] bool, pos_padded [B, max_pos] long,
        pos_counts [B] long) for a batch of user ids."""
        B = len(subs)
        known_tensors = [self._known_items_local_per_user[int(u)] for u in subs]
        test_tensors = [self._test_items_local_per_user[int(u)] for u in subs]

        known_counts = torch.tensor([t.numel() for t in known_tensors], dtype=torch.long)
        test_counts_cpu = torch.tensor([t.numel() for t in test_tensors], dtype=torch.long)

        known_mask = torch.zeros(B, self.n_items, dtype=torch.bool)
        if int(known_counts.sum()) > 0:
            known_flat = torch.cat(known_tensors)
            known_rows = torch.repeat_interleave(torch.arange(B, dtype=torch.long), known_counts)
            known_mask[known_rows, known_flat] = True

        max_pos = int(test_counts_cpu.max().item()) if B > 0 else 0
        pos_padded = torch.full((B, max(max_pos, 1)), -1, dtype=torch.long)
        if max_pos > 0 and int(test_counts_cpu.sum()) > 0:
            test_flat = torch.cat(test_tensors)
            test_rows = torch.repeat_interleave(torch.arange(B, dtype=torch.long), test_counts_cpu)
            offsets = test_counts_cpu.cumsum(0) - test_counts_cpu
            slot = torch.arange(test_flat.numel(), dtype=torch.long) - offsets[test_rows]
            pos_padded[test_rows, slot] = test_flat

        return (
            known_mask.to(device, non_blocking=True),
            pos_padded.to(device, non_blocking=True),
            test_counts_cpu.to(device, non_blocking=True),
        )

    # ------------------------------------------------------------------
    # batch helpers
    # ------------------------------------------------------------------
    def get_batch(self, batch_idx, data='train'):
        if data == 'train':
            query  = np.array(self.train_q)
            answer = self.train_a
            wrongs = self.train_w
            subs = query[batch_idx, 0]
            rels = query[batch_idx, 1]
            pos = answer[batch_idx[0]:batch_idx[-1] + 1]
            neg = wrongs[batch_idx[0]:batch_idx[-1] + 1]
            return subs, rels, pos, neg
        else:
            query  = np.array(self.test_q)
            answer = np.array(self.test_a, dtype=object)
            subs = query[batch_idx, 0]
            rels = query[batch_idx, 1]
            objs = np.zeros((len(batch_idx), self.n_nodes))
            for i in range(len(batch_idx)):
                objs[i][answer[batch_idx[i]]] = 1
            return subs, rels, objs

    # ------------------------------------------------------------------
    # shuffle
    # ------------------------------------------------------------------
    def shuffle_train(self):
        if 'Dis_5fold_item' in self.task_dir:
            self.facts_cf, self.train_cf = self.generate_inductive_train(self.all_cf)
            self.fact_triple  = self.cf_to_triple(self.facts_cf)
            self.train_triple = self.cf_to_triple(self.train_cf)
            self.fact_data, _ = self.interact_triple(self.d_triple)
            if self.readukg:
                self.fact_data += self.ukg
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        elif any(tag in self.task_dir for tag in
                 ['new_last-fm', 'new_amazon-book', 'new_alibaba-fashion']):
            self.train_triple = np.array(self.train_triple)
            rand_idx = np.random.permutation(len(self.train_triple))
            self.train_triple = self.train_triple[rand_idx].tolist()
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
        else:
            fact_triple  = np.array(self.fact_triple)
            train_triple = np.array(self.train_triple)
            all_ui = np.concatenate([fact_triple, train_triple], axis=0)
            rand_idx = np.random.permutation(len(all_ui))
            all_ui = all_ui[rand_idx]
            split = len(all_ui) * 6 // 7
            self.fact_triple  = all_ui[:split].tolist()
            self.train_triple = all_ui[split:].tolist()
            self.fact_data, _ = self.interact_triple(self.d_triple)
            if self.readukg:
                self.fact_data += self.ukg
            self.load_graph(self.fact_data)
            self.train_q, self.train_a, self.train_w = self.load_train_query(self.train_triple)
            self.n_train = len(self.train_q)
