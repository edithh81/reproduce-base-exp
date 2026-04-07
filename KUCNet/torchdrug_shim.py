"""Minimal shim for torchdrug.layers.functional.variadic_topk (Python 3.12 compat)."""
import torch


def variadic_topk(input, size, k, largest=True):
    """
    Return the top-k values and indices for variadic-length groups.

    Args:
        input: 1-D tensor of values
        size: 1-D tensor of group sizes (sums to len(input))
        k: number of top elements per group
        largest: if True, return largest; else smallest

    Returns:
        (values, indices) — both shaped (n_groups * k,)
    """
    starts = torch.zeros_like(size)
    starts[1:] = size[:-1].cumsum(0)

    all_values = []
    all_indices = []
    for i in range(len(size)):
        s = starts[i].item()
        n = size[i].item()
        group = input[s:s + n]
        actual_k = min(k, n)
        if actual_k == 0:
            vals = input.new_zeros(k)
            idxs = input.new_zeros(k, dtype=torch.long)
        else:
            vals, idxs = torch.topk(group, actual_k, largest=largest)
            if actual_k < k:
                pad_v = input.new_zeros(k - actual_k)
                pad_i = input.new_zeros(k - actual_k, dtype=torch.long)
                vals = torch.cat([vals, pad_v])
                idxs = torch.cat([idxs, pad_i])
            idxs = idxs + s  # global indices
        all_values.append(vals)
        all_indices.append(idxs)

    return torch.cat(all_values), torch.cat(all_indices)
