import os
import random

import numpy as np
import torch


def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, torch (CPU + all CUDA devices) and optionally
    force deterministic algorithms / cuDNN for bit-for-bit reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[seed_everything] use_deterministic_algorithms failed ({e})")
    else:
        torch.backends.cudnn.benchmark = True


def cal_bpr_loss(n_users, pos, neg, scores):
    """BPR loss for recommendation.
    pos/neg: lists of arrays, each array contains item indices (global node ids).
    scores: [batch, n_items] tensor (item scores, indexed from 0).
    """
    n = scores.shape[0]
    loss = 0
    for i in range(n):
        pos_score = scores[i][pos[i] - n_users]
        neg_score = scores[i][neg[i] - n_users]
        u_loss = -1 * torch.sum(torch.nn.LogSigmoid()(pos_score - neg_score))
        loss += u_loss
    return loss


def ndcg_k(r, k, len_pos_test):
    if len_pos_test > k:
        standard = [1.0] * k
    else:
        standard = [1.0] * len_pos_test + [0.0] * (k - len_pos_test)
    dcg_max = dcg_k(standard, k)
    if dcg_max == 0:
        return 0.0
    return dcg_k(r, k) / dcg_max


def dcg_k(r, k):
    r = np.asarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def _build_hit_matrix(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """[B, K] float hit indicator — 1.0 if rank k matches a valid positive."""
    match = topk_idx.unsqueeze(-1) == pos_padded.unsqueeze(1)  # [B, K, P]
    p = pos_padded.size(1)
    pos_mask = torch.arange(p, device=topk_idx.device).unsqueeze(0) < pos_counts.unsqueeze(1)
    match = match & pos_mask.unsqueeze(1)
    return match.any(dim=-1).float()


def recall_at_k(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """Per-user recall@K. Returns [B] float tensor."""
    hits = _build_hit_matrix(topk_idx, pos_padded, pos_counts)
    denom = pos_counts.clamp(min=1).to(hits.dtype)
    return hits.sum(dim=-1) / denom


def ndcg_at_k(topk_idx: torch.Tensor, pos_padded: torch.Tensor, pos_counts: torch.Tensor) -> torch.Tensor:
    """Per-user nDCG@K matching the ndcg_k/dcg_k definition above."""
    hits = _build_hit_matrix(topk_idx, pos_padded, pos_counts)
    K = topk_idx.size(1)
    discount = 1.0 / torch.log2(
        torch.arange(2, K + 2, device=topk_idx.device, dtype=hits.dtype)
    )
    dcg = (hits * discount.unsqueeze(0)).sum(dim=-1)

    ideal_len = pos_counts.clamp(max=K).to(hits.dtype)
    ideal_mask = (
        torch.arange(K, device=topk_idx.device, dtype=hits.dtype).unsqueeze(0)
        < ideal_len.unsqueeze(1)
    ).to(hits.dtype)
    idcg = (ideal_mask * discount.unsqueeze(0)).sum(dim=-1)
    return dcg / idcg.clamp(min=1e-12)
