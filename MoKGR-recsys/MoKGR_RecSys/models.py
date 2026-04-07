"""
MoKGR model adapted for Recommendation.

Core change vs. the original transductive MoKGR:
  - The graph now contains *users* (0 ... n_users-1) **and** *items/KG entities*
    (n_users ... n_nodes-1).
  - Scoring is over **items only** (n_items dimensional output).
  - Relation embedding size accounts for interact / inv-interact rels and
    the self-loop relation added by the data loader.

Everything else (MoE for hops, MoE for pruning, Gumbel-Sigmoid gate,
GRU gating, GNN message passing) is identical to the original MoKGR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch_scatter
from moe import MoE_for_hops, MoE_for_Pruning


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_nodes, n_users, n_items, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.n_nodes = n_nodes
        self.n_users = n_users
        self.n_items = n_items
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        # +3 for: interact(0), inv-interact(1), KG rels shifted by 2, inverses, self-loop
        self.rela_embed = nn.Embedding(2 * n_rel + 3, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        message = hs + hr
        alpha_temp = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        alpha = torch.sigmoid(alpha_temp)
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new, alpha, message, obj, alpha_temp, self.act


class MoKGR_RecSys(torch.nn.Module):
    def __init__(self, params, loader):
        super(MoKGR_RecSys, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        self.max_hop = params.max_hop
        self.min_hop = params.min_hop
        self.hop_temperature = params.hop_temperature
        self.pruning_temperature = params.pruning_temperature
        self.lambda_noise = params.lambda_noise
        self.K_source = params.K_source
        self.K_min = params.K_min
        self.K_max = params.K_max
        self.l_inflection = params.l_inflection
        self.a = params.a
        self.num_pruning_experts = params.num_pruning_experts
        self.tau = params.tau
        self.lambda_noise_pruning = params.lambda_noise_pruning

        # RecSys-specific: unified graph dimensions
        self.n_ent = loader.n_ent
        self.n_nodes = loader.n_nodes
        self.n_users = loader.n_users
        self.n_items = loader.n_items

        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(
                self.hidden_dim, self.hidden_dim, self.attn_dim,
                self.n_rel, self.n_nodes, self.n_users, self.n_items,
                act=act
            ))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.gumbel_temp = 1.0

        # MoE for hops
        self.moe_for_hops = MoE_for_hops(
            loader,
            hidden_dim=self.hidden_dim,
            num_experts=params.num_experts,
            min_hop=self.min_hop,
            max_hop=self.max_hop,
            lambda_noise=self.lambda_noise,
            temperature=self.hop_temperature
        )

        # MoE for pruning - uses n_nodes (unified graph size)
        self.MoE_for_Pruning = MoE_for_Pruning(
            K_source=self.K_source,
            K_min=self.K_min,
            K_max=self.K_max,
            l_inflection=self.l_inflection,
            a=self.a,
            n_rel=self.n_rel,
            n_ent=self.n_nodes,  # Key change: use n_nodes instead of n_ent
            hidden_dim=self.hidden_dim,
            tau=self.tau,
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_pruning_experts=self.num_pruning_experts,
            temperature=self.pruning_temperature,
            lambda_noise_pruning=self.lambda_noise_pruning
        )

        self.layer_embed = nn.Embedding(params.n_layer, params.hidden_dim)

        self.gate_threshold = params.gate_threshold
        self.active_gate = params.active_gate
        self.G_full = None
        self.Q = None

    def forward(self, subs, rels, mode='train'):
        self.G_full = None
        self.Q = None
        n = len(subs)

        L_importance = 0
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        # Item-only scores: shape (batch, n_items)
        scores_all = torch.zeros((n, self.n_items)).cuda()
        j = 0
        for i in range(self.n_layer):

            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)

            hidden, alpha, message, obj, alpha_temp, act = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0))
            if self.active_gate:
                mu_l = hidden.mean(dim=0)
                sigma_l = hidden.std(dim=0)

                gate_input = torch.cat([
                    mu_l,
                    sigma_l
                ], dim=-1)

                gate_logit = self.gate_mlp(gate_input)

                # Gating with Gumbel-Sigmoid
                if mode == 'train':
                    gate_value = torch.sigmoid((gate_logit + sample_gumbel(gate_logit.shape).cuda()) / self.gumbel_temp)
                else:
                    if i >= self.n_layer // 2:
                        cv_l = sigma_l.abs() / (mu_l.abs() + 1e-8)
                        gate_value = (cv_l.mean() <= self.gate_threshold).float()
                    else:
                        gate_value = torch.ones_like(gate_logit)

                hidden = hidden * gate_value

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            if i >= self.min_hop - 1 and i <= self.max_hop - 1:

                if i == self.min_hop - 1:
                    self.G_full, self.Q = self.moe_for_hops(subs, rels, hidden)
                if self.G_full is None:
                    raise ValueError(
                        f"G_full is None at layer {i}. Check if min_hop ({self.min_hop}) is correctly set.")
                scores = self.W_final(hidden).squeeze(-1)
                hidden, nodes, scores, h0, L_importance_temp = self.MoE_for_Pruning(
                    hidden, nodes, scores, h0, alpha, hidden, q_rel,
                    edges, old_nodes_new_idx, n, message, obj, alpha_temp, act, (i + 1)
                )

                L_importance = L_importance + L_importance_temp

                # Apply hop weight and accumulate item-only scores
                weighted_scores = scores * self.G_full[j]
                item_mask = (nodes[:, 1] >= self.n_users) & (nodes[:, 1] < self.n_users + self.n_items)
                if item_mask.any():
                    item_nodes = nodes[item_mask]
                    item_scores = weighted_scores[item_mask]
                    scores_all[item_nodes[:, 0], item_nodes[:, 1] - self.n_users] += item_scores
                j += 1

            if self.active_gate:
                if gate_value.item() == 0:
                    break

        if mode == 'train':
            return scores_all, self.G_full, self.Q, L_importance
        else:
            return scores_all


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)
