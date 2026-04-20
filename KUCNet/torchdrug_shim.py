"""Minimal shim for torchdrug.layers.functional.variadic_topk (Python 3.12 compat).

Vectorized rewrite: groups are padded into a dense [n_groups, padded_len]
matrix and a single batched ``torch.topk`` replaces the per-group Python loop.
"""
import torch


def variadic_topk(input, size, k, largest=True):
    """Return the top-k values and indices for variadic-length groups.

    Args:
        input:   1-D tensor of values, groups concatenated contiguously.
        size:    1-D LongTensor of group sizes (sums to ``len(input)``).
        k:       number of top elements per group.
        largest: if True, return largest; else smallest.

    Returns:
        values, indices — both shaped ``(n_groups * k,)``.
        ``indices`` are global indices into ``input`` (i.e. include the
        per-group start offset), matching the previous looped shim.

        For groups shorter than ``k`` the trailing slots are zero-valued
        and point at the group's start index; for empty groups both value
        and index slots are zero (also matching the previous behavior).
    """
    n_groups = size.numel()
    device = input.device

    if n_groups == 0:
        return (
            input.new_zeros(0),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    max_count = int(size.max().item())
    padded_len = max(max_count, k)

    starts = torch.zeros_like(size)
    if n_groups > 1:
        starts[1:] = size[:-1].cumsum(0)

    # Scatter ``input`` into a dense [n_groups, padded_len] matrix. Empty
    # slots are filled with ±inf so ``torch.topk`` ignores them.
    pad_fill = float("-inf") if largest else float("inf")
    padded = input.new_full((n_groups, padded_len), pad_fill)
    if input.numel() > 0:
        group_idx = torch.repeat_interleave(
            torch.arange(n_groups, device=device), size
        )
        pos_in_group = (
            torch.arange(input.numel(), device=device) - starts[group_idx]
        )
        padded[group_idx, pos_in_group] = input

    values, local_idx = torch.topk(padded, k, dim=1, largest=largest)

    # Slots beyond a group's true length are padding — zero their value
    # and (local) index so the globalized index becomes ``starts[g]``.
    valid = local_idx < size.unsqueeze(1)
    values = torch.where(valid, values, torch.zeros_like(values))
    local_idx = torch.where(valid, local_idx, torch.zeros_like(local_idx))

    global_idx = local_idx + starts.unsqueeze(1)

    # Empty groups: original loop returned all-zero indices (no offset add);
    # preserve that to stay bug-for-bug compatible.
    empty = (size == 0).unsqueeze(1)
    if empty.any():
        global_idx = torch.where(empty, torch.zeros_like(global_idx), global_idx)

    return values.reshape(-1), global_idx.reshape(-1)
