import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter, scatter_max

class MoE_for_Pruning(nn.Module):
    def __init__(self, K_source, K_min, K_max, l_inflection, a, n_rel, n_ent, hidden_dim, tau, in_dim, out_dim, num_pruning_experts, temperature, lambda_noise_pruning=1.0):
        """
        n_ent: total number of nodes in the unified graph (n_users + n_kg_entities).
        n_rel: number of original KG relations (before doubling/shifting).
        """
        super(MoE_for_Pruning, self).__init__()
        self.K_source = K_source
        self.K_min = K_min
        self.K_max = K_max
        self.l_inflection = l_inflection
        self.a = a
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_ent = n_ent
        self.temperature = temperature
        self.lambda_noise_pruning = lambda_noise_pruning

        if self.training and self.tau > 0:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)
        self.num_pruning_experts = num_pruning_experts
        assert 1 <= self.num_pruning_experts <= 3, "num_pruning_experts must be between 1 and 3"

        # Define the MLP for generating expert weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.sub_embed = nn.Embedding(n_ent, in_dim)
        # +3 for: interact(0), inv-interact(1), KG rels shifted by 2, inverses, self-loop
        self.rel_embed = nn.Embedding(2 * n_rel + 3, in_dim)

        self.expert_embedding = nn.Parameter(torch.randn(3, hidden_dim))

        # Initialize the pruning experts
        self.W_h_alpha = nn.Linear(in_dim, out_dim, bias=False)  # For Alpha_Pruning
        self.W_h_similarity = nn.Linear(in_dim, out_dim, bias=False)  # For Similarity_Pruning

        # Linear layer for the noise term
        self.W_n = nn.Linear(hidden_dim, 1, bias=False)

        # Softplus activation function
        self.softplus = nn.Softplus()

    def forward(self, hidden, nodes, scores, h0, alpha, hidden_q, q_rel, edges, old_nodes_new_idx, batch_size, message, obj, alpha_temp, act, l):
        n_node = nodes.size(0)
        n_ent = self.n_ent
        hidden_dim = hidden.size(1)

        q_rel_embed = self.rel_embed(q_rel)  # [batch_size, in_dim]

        node_batch_indices = nodes[:, 0].long()
        hidden_q_expanded = hidden_q[node_batch_indices, :]
        q_rel_embed_expanded = q_rel_embed[node_batch_indices, :]
        # Generate context vector
        context_input = torch.cat([hidden_q_expanded, q_rel_embed_expanded], dim=-1)
        context_vector = self.weight_mlp(context_input)

        # Calculate expert scores via dot product
        expert_scores = torch.matmul(context_vector, self.expert_embedding.t())

        # Add noise
        noise_scale = self.softplus(self.W_n(context_vector))
        noise = torch.randn_like(expert_scores)
        expert_scores = expert_scores + self.lambda_noise_pruning * noise * noise_scale
        weights = F.softmax(expert_scores / self.temperature, dim=-1)

        # Compute importance per expert
        importance = weights.sum(dim=0) / weights.sum()

        # Select the top num_pruning_experts weights and their indices
        topk_weights, topk_indices = torch.topk(importance, self.num_pruning_experts)
        selected_weights = topk_weights / topk_weights.sum()

        selected_indices = topk_indices.tolist()

        device = hidden.device

        # Identify new nodes (nodes not in old_nodes_new_idx)
        tmp_diff_node_idx = torch.ones(n_node, dtype=torch.bool, device=device)
        tmp_diff_node_idx[old_nodes_new_idx] = False
        bool_diff_node_idx = tmp_diff_node_idx

        diff_node = nodes[bool_diff_node_idx]
        hidden_diff = hidden[bool_diff_node_idx]
        scores_diff = scores[bool_diff_node_idx]
        h0_diff = h0[:, bool_diff_node_idx, :]

        # Initialize combined outputs
        all_nodes = nodes
        n_all_nodes = all_nodes.size(0)
        hidden_combined = torch.zeros((n_all_nodes, hidden_dim), device=device)
        scores_combined = torch.zeros((n_all_nodes,), device=device)
        h0_combined = torch.zeros((1, n_all_nodes, hidden_dim), device=device)

        # Create a node mapping
        node_ids = all_nodes[:, 0] * n_ent + all_nodes[:, 1]
        max_node_id = node_ids.max().item()
        node_id_to_index_tensor = torch.full((max_node_id + 1,), -1, dtype=torch.long, device=device)
        node_id_to_index_tensor[node_ids] = torch.arange(n_all_nodes, device=device)

        # Calculate the Top-K value for the current layer
        K_l = self.compute_topk(l)

        for idx, weight in zip(selected_indices, selected_weights):
            if idx == 0:
                hidden_i, nodes_i, scores_i, h0_i = self.nodes_pruner(
                    hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                    hidden_diff, scores_diff, h0_diff, batch_size, K_l
                )
            elif idx == 1:
                hidden_i, nodes_i, scores_i, h0_i = self.similarity_pruner(
                    hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                    hidden_diff, scores_diff, h0_diff, q_rel_embed, batch_size, K_l
                )
            elif idx == 2:
                hidden_i, nodes_i, scores_i, h0_i = self.alpha_pruner(
                    hidden, nodes, scores, h0, edges, alpha_temp, message, obj,
                    act, bool_diff_node_idx, old_nodes_new_idx, n_node, K_l
                )
            else:
                continue

            node_ids_i = nodes_i[:, 0] * n_ent + nodes_i[:, 1]
            indices_in_all = node_id_to_index_tensor[node_ids_i]

            hidden_combined.index_add_(0, indices_in_all, weight * hidden_i)
            scores_combined.index_add_(0, indices_in_all, weight * scores_i)
            h0_combined.index_add_(1, indices_in_all, weight * h0_i)

        # Filter out nodes with zero scores
        mask = scores_combined != 0
        hidden_combined = hidden_combined[mask]
        nodes = all_nodes[mask]
        scores_combined = scores_combined[mask]
        h0_combined = h0_combined[:, mask, :]

        L_importance = self.compute_importance_loss(weights)

        return hidden_combined, nodes, scores_combined, h0_combined, L_importance

    def compute_importance_loss(self, importance):
        mean_importance = importance.mean()
        std_importance = importance.std()
        CV_importance = std_importance / (mean_importance + 1e-5)
        L_importance = CV_importance ** 2
        return L_importance

    def compute_topk(self, l):
        if l < self.l_inflection:
            S_l = 1 / (1 + math.exp(-self.a * (l - self.l_inflection / 2)))
            K_l = self.K_source + (self.K_max - self.K_source) * S_l
        else:
            S_l = 1 / (1 + math.exp(-self.a * (l - 3 * self.l_inflection / 2)))
            K_l = self.K_min + (self.K_max - self.K_min) * (1 - S_l)
        return int(K_l)

    def nodes_pruner(self, hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                     hidden_diff, scores_diff, h0_diff, batch_size, K_l):
        n_ent = self.n_ent
        device = hidden.device

        node_scores = torch.full((batch_size, n_ent), float('-inf'), device=device)
        node_scores[diff_node[:, 0], diff_node[:, 1]] = scores_diff

        node_probs = self.softmax(node_scores)

        topk_probs, topk_indices = torch.topk(node_probs, K_l, dim=1)
        topk_mask = torch.zeros_like(node_probs).scatter_(1, topk_indices, 1).bool()

        bool_sampled_diff_nodes_idx = topk_mask[diff_node[:, 0], diff_node[:, 1]]

        diff_node_prob = node_probs[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob_hard = topk_mask[diff_node[:, 0], diff_node[:, 1]].float()

        hidden_updated = hidden.clone()
        hidden_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        h0_updated = h0.clone()
        h0_updated[:, bool_diff_node_idx, :] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(0).unsqueeze(-1)
        scores_updated = scores.clone()
        scores_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob)

        return hidden_updated, nodes, scores_updated, h0_updated

    def similarity_pruner(self, hidden, nodes, scores, h0, bool_diff_node_idx, diff_node,
                          hidden_diff, scores_diff, h0_diff, q_rel_embed, batch_size, K_l):
        n_ent = self.n_ent
        device = hidden.device

        node_batch_indices = diff_node[:, 0]
        node_q_rel_embed = q_rel_embed[node_batch_indices]

        similarities = F.cosine_similarity(hidden_diff, node_q_rel_embed, dim=1)

        similarity_scores = torch.full((batch_size, n_ent), float('-inf'), device=device)
        similarity_scores[diff_node[:, 0], diff_node[:, 1]] = similarities

        node_probs = self.softmax(similarity_scores)

        topk_probs, topk_indices = torch.topk(node_probs, K_l, dim=1)
        topk_mask = torch.zeros_like(node_probs).scatter_(1, topk_indices, 1).bool()

        bool_sampled_diff_nodes_idx = topk_mask[diff_node[:, 0], diff_node[:, 1]]

        diff_node_prob = node_probs[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob_hard = topk_mask[diff_node[:, 0], diff_node[:, 1]].float()

        hidden_updated = hidden.clone()
        hidden_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        h0_updated = h0.clone()
        h0_updated[:, bool_diff_node_idx, :] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(0).unsqueeze(-1)
        scores_updated = scores.clone()
        scores_updated[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob)

        return hidden_updated, nodes, scores_updated, h0_updated

    def alpha_pruner(self, hidden, nodes, scores, h0, edges, alpha_temp, message, obj,
                     act, bool_diff_node_idx, old_nodes_new_idx, n_node, K_l):
        device = hidden.device

        n_edges = edges.size(0)
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj_indices = edges[:, 5]

        node_is_new = bool_diff_node_idx
        edge_target_is_new = node_is_new[obj_indices]

        alpha = alpha_temp.squeeze(-1)

        max_alpha_per_node, _ = scatter_max(
            alpha,
            obj_indices,
            dim=0,
            dim_size=n_node
        )

        node_probs = torch.softmax(max_alpha_per_node / self.temperature, dim=0)

        new_nodes_indices = torch.nonzero(bool_diff_node_idx).squeeze(-1)
        if new_nodes_indices.numel() > 0:
            new_nodes_probs = node_probs[new_nodes_indices]

            topk = min(K_l, new_nodes_indices.numel())
            topk_values, topk_indices = torch.topk(new_nodes_probs, topk)
            selected_new_nodes = new_nodes_indices[topk_indices]

            keep_nodes = torch.zeros(n_node, dtype=torch.bool, device=device)
            keep_nodes[selected_new_nodes] = True
            keep_nodes[~bool_diff_node_idx] = True
        else:
            keep_nodes = ~bool_diff_node_idx

        alpha = torch.sigmoid(alpha).unsqueeze(-1)
        message = alpha * message

        message_agg = scatter(message, index=obj_indices, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = act(self.W_h_alpha(message_agg))

        hidden_updated = hidden_new[keep_nodes]
        nodes_updated = nodes[keep_nodes]
        scores_updated = scores[keep_nodes]
        h0_updated = h0[:, keep_nodes, :]

        return hidden_updated, nodes_updated, scores_updated, h0_updated


class MoE_for_hops(nn.Module):
    def __init__(self, loader, hidden_dim, num_experts, min_hop, max_hop, lambda_noise=1.0, temperature=1.0):
        super(MoE_for_hops, self).__init__()
        self.num_experts = num_experts
        self.min_hop = min_hop
        self.max_hop = max_hop
        self.hop_range = max_hop - min_hop + 1
        self.lambda_noise = lambda_noise
        self.embed_dim = hidden_dim
        self.temperature = temperature

        # MLP: Generate contextual embedding c_i from [h_i, h_q]
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # The embedding for each hop expert
        self.hop_embedding = nn.Embedding(self.hop_range, hidden_dim)
        # Use n_nodes for entity embeddings (unified user+entity graph)
        self.entity_embed = nn.Embedding(loader.n_nodes, self.embed_dim)
        # +3 for interact(0), inv-interact(1), KG rels shifted, self-loop
        self.relation_embed = nn.Embedding(loader.n_rel * 2 + 3, self.embed_dim)

        # Linear layer W_m for load balancing
        self.W_m = nn.Linear(1, 1, bias=False)

        # Linear layer for the noise term
        self.W_n = nn.Linear(hidden_dim, 1, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, subs, rels, hidden):
        q_rel = torch.LongTensor(rels).cuda()
        h_rq = self.relation_embed(q_rel)

        batch_size = len(subs)
        indices = torch.arange(batch_size, device=hidden.device)
        h_rq_min = hidden[indices]

        mlp_input = torch.cat([h_rq_min, h_rq], dim=-1)
        c_i = self.context_mlp(mlp_input)
        c_i = torch.mean(c_i, dim=0, keepdim=True)

        hops = torch.arange(self.min_hop, self.max_hop + 1).cuda()
        phi_hop = self.hop_embedding(hops - self.min_hop)

        Q = torch.matmul(c_i, phi_hop.t()).squeeze(0)

        noise = torch.randn_like(Q) * self.softplus(self.W_n(c_i)).squeeze(0)
        Q = Q + self.lambda_noise * noise

        topk_values, topk_indices = torch.topk(Q, self.num_experts, dim=0)

        G_topk = F.softmax(topk_values / self.temperature, dim=0)

        G_full = torch.zeros(self.hop_range).cuda()
        G_full[topk_indices] = G_topk

        return G_full, Q

    def compute_importance_loss(self, G_full):
        importance = G_full
        mean_importance = importance.mean()
        std_importance = importance.std()
        CV_importance = std_importance / (mean_importance + 1e-5)
        L_importance = CV_importance ** 2
        return L_importance

    def compute_load_loss(self, Q, G_full):
        kth_ex = Q.topk(self.num_experts, dim=0, largest=True, sorted=True)[0][-1]

        sigma_hop = self.softplus(self.W_m(Q.unsqueeze(1))).squeeze(1)

        P_hi_o = 0.5 * (1 + torch.erf((Q - kth_ex) / (sigma_hop * math.sqrt(2))))

        mean_p = P_hi_o.mean()
        std_p = P_hi_o.std()
        CV_load = std_p / (mean_p + 1e-8)
        L_load = CV_load ** 2
        return L_load
