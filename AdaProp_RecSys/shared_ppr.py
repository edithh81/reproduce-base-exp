"""
Shared per-user PPR for fair cross-codebase benchmarking.

One set of PPR scores (alpha=0.85, beta=0.8, N=20 fixed iterations) computed
once per dataset and reused across codebases that share the same per-user
teleport semantics (KISS, KUCNet, AdaProp_RecSys). MoKGR uses a per-node
PPRSampler with different semantics and is deliberately excluded.

Cache layout (resolved from this file's location, independent of cwd):
    <repo-root>/shared_ppr_cache/<dataset>_user_ppr_top{K}.pt

Cache payload is a dict on CPU:
    {
        'mode'     : 'topk',
        'indices'  : LongTensor  [n_users, K]  (node ids),
        'scores'   : FloatTensor [n_users, K]  (PPR scores, float16),
        'ppr_topk' : int,
    }

Design notes:
- No `tol`-based early termination; fixed N=20 iterations for bit-for-bit
  reproducibility across codebases that read the same cache.
- Transition matrix follows the KUCNet/AdaProp convention
  (M = Mkg @ diag(1/out_deg)); the row of M that is indexed by column gives
  the in-edges weighted by the tail's out-degree. Kept as-is so cached
  scores match the original papers' reported setup.
"""

import os
import time

import torch
from tqdm import tqdm


_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "shared_ppr_cache",
)


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cpu")
    dev = str(device).lower()
    if dev in {"auto", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[shared_ppr] CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dataset_name(loader, dataset_name):
    if dataset_name:
        return str(dataset_name).strip().strip("/").replace("/", "_")
    td = getattr(loader, "task_dir", None)
    if td is None:
        raise ValueError(
            "shared_ppr: cannot derive dataset name; pass dataset_name="
            "<dataset> or set loader.task_dir."
        )
    base = str(td).strip().strip("/")
    return os.path.basename(base) or base.replace("/", "_")


def _compute_ppr(
    loader,
    topk,
    bs=128,
    N=20,
    alpha=0.85,
    beta=0.8,
    compute_dtype=torch.float32,
    cache_dtype=torch.float16,
    device=None,
):
    device = _normalize_device(device if device is not None else "auto")
    n_nodes = int(loader.n_nodes)
    n_users = int(loader.n_users)
    k = min(int(topk), n_nodes)

    tkg = torch.as_tensor(loader.tKG, dtype=torch.long, device=device)
    heads = tkg[:, 0]
    tails = tkg[:, 2]

    out_deg = torch.bincount(heads, minlength=n_nodes).clamp_min(1)
    values = (1.0 / out_deg[tails]).to(dtype=compute_dtype)
    indices = torch.stack([heads, tails], dim=0)
    M = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_nodes), dtype=compute_dtype, device=device
    ).coalesce()

    final_indices = torch.empty((n_users, k), dtype=torch.long)
    final_scores = torch.empty((n_users, k), dtype=cache_dtype)

    n_batch = n_users // bs + int(n_users % bs > 0)
    s_time = time.time()

    for i in tqdm(range(n_batch), desc=f"[shared_ppr] compute on {device.type}"):
        start = i * bs
        end = min(n_users, start + bs)
        tbs = end - start
        u_list = list(range(start, end))

        rank = torch.zeros((n_nodes, tbs), dtype=compute_dtype, device=device)
        col_idx = torch.arange(tbs, dtype=torch.long, device=device)
        user_idx = torch.as_tensor(u_list, dtype=torch.long, device=device)
        rank[user_idx, col_idx] = 1.0

        P = torch.empty((n_nodes, tbs), dtype=compute_dtype, device=device)
        for col, uid in enumerate(u_list):
            p_set = loader.known_user_set[uid]
            n_pref = len(p_set)

            if n_pref >= n_nodes:
                P[:, col].fill_(1.0 / n_nodes)
                continue

            denom = max(1, n_nodes - n_pref)
            base = (1.0 - beta) / denom
            P[:, col].fill_(base)

            if n_pref > 0:
                pref_idx = torch.as_tensor(
                    list(p_set), dtype=torch.long, device=device
                )
                P[pref_idx, col] = beta / n_pref

        for _ in range(N):
            rank = (1.0 - alpha) * P + alpha * torch.sparse.mm(M, rank)

        rank_t = rank.transpose(0, 1).cpu().float()  # [tbs, n_nodes]
        scores, idx = torch.topk(rank_t, k=k, dim=1)
        final_indices[start:end] = idx
        final_scores[start:end] = scores.to(dtype=cache_dtype)

        if device.type == "cuda":
            del rank, P, rank_t
            torch.cuda.empty_cache()

    print(f"[shared_ppr] done on {device.type}. time: {time.time() - s_time:.1f}s")
    return final_indices.contiguous(), final_scores.contiguous()


def get_shared_ppr(
    loader,
    topk,
    dataset_name=None,
    bs=128,
    N=20,
    alpha=0.85,
    beta=0.8,
    device="auto",
    fallback_to_cpu=True,
    cache_dir=None,
):
    """Return (indices, scores) as CPU tensors, computing + caching if needed.

    Parameters
    ----------
    loader : object exposing .tKG, .n_nodes, .n_users, .known_user_set, .task_dir
    topk   : int, number of top PPR nodes retained per user
    dataset_name : optional override for cache filename; defaults to
        basename(loader.task_dir)
    bs, N, alpha, beta : PPR hyperparameters (fixed defaults match KISS/KUCNet/
        AdaProp reported setup)
    device : 'auto' | 'cuda' | 'cpu' | torch.device
    fallback_to_cpu : retry on CPU if CUDA preprocessing OOMs
    cache_dir : optional override; default <repo-root>/shared_ppr_cache/

    Returns
    -------
    indices : LongTensor  [n_users, topk]  (CPU)
    scores  : FloatTensor [n_users, topk]  (CPU, float16 cached → upcast here)
    """
    n_nodes = int(loader.n_nodes)
    n_users = int(loader.n_users)
    k = min(int(topk), n_nodes)
    name = _resolve_dataset_name(loader, dataset_name)

    if cache_dir is None:
        cache_dir = _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{name}_user_ppr_top{k}.pt")

    if os.path.exists(cache_path):
        print(f"[shared_ppr] loading cached scores from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu")
        ind = blob["indices"]
        sc = blob["scores"]
        if ind.shape == (n_users, k) and sc.shape == (n_users, k):
            return ind.long().contiguous(), sc.float().contiguous()
        print(
            f"[shared_ppr] cache shape mismatch (got {tuple(ind.shape)}, expected "
            f"({n_users}, {k})); recomputing."
        )

    try:
        ind, sc = _compute_ppr(
            loader, topk=k, bs=bs, N=N, alpha=alpha, beta=beta, device=device
        )
    except RuntimeError as e:
        msg = str(e).lower()
        oom_like = any(
            t in msg for t in ["out of memory", "cuda error", "cublas", "cusparse"]
        )
        cur = _normalize_device(device)
        if cur.type == "cuda" and fallback_to_cpu and oom_like:
            print(f"[shared_ppr] GPU failed ({e}); falling back to CPU once.")
            torch.cuda.empty_cache()
            ind, sc = _compute_ppr(
                loader,
                topk=k,
                bs=bs,
                N=N,
                alpha=alpha,
                beta=beta,
                device=torch.device("cpu"),
            )
        else:
            raise

    torch.save(
        {"mode": "topk", "indices": ind, "scores": sc, "ppr_topk": int(k)},
        cache_path,
    )
    print(
        f"[shared_ppr] cached to {cache_path} "
        f"({os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)"
    )
    return ind.long().contiguous(), sc.float().contiguous()
