#!/usr/bin/env python3
"""
Subset Knowledge Graph Recommendation Datasets
================================================
Creates smaller subsets of KGRS datasets while preserving key statistical
characteristics of the originals:

1. User/Item interaction density distributions (degree distributions)
2. Train/test split ratio
3. KG connectivity: multi-hop neighborhood of subset items is retained
4. Relation type distribution in KG triples
5. Sparsity level: interactions / (users × items)
6. User coverage: test users must exist in train

Strategy (Importance Sampling → BFS KG Expansion):
----------------------------------------------------
Step 1 — Stratified User Sampling:
    Users are bucketed by their interaction count (degree). We sample
    proportionally from each bucket so the per-user degree distribution
    in the subset mirrors the original.

Step 2 — Item Retention:
    Keep only items that appear in the sampled users' interactions.
    This naturally preserves the item popularity (long-tail) distribution
    because high-degree users interact with popular items, and the
    proportional user sampling maintains the mix.

Step 3 — KG Subgraph Extraction (BFS):
    Starting from the retained item set, perform a K-hop BFS over the
    original KG to collect all reachable entities and triples. This
    preserves the local KG structure around items, which is exactly
    what the GNN message-passing in the model uses.

Step 4 — ID Remapping:
    Remap all user, item, entity, and relation IDs to be contiguous
    starting from 0, and regenerate all auxiliary files (entity_list,
    item_list, user_list, relation_list).

Step 5 — Validation Report:
    Print a side-by-side comparison of original vs subset statistics
    as evidence that characteristics are preserved.
"""

import os
import sys
import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# ──────────────────────────── I/O helpers ────────────────────────────

def read_cf(filepath):
    """Read user-item interaction file. Each line: user_id item1 item2 ..."""
    user_items = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            uid, items = parts[0], parts[1:]
            user_items[uid].extend(items)
    return user_items  # {uid: [item_ids]}


def write_cf(filepath, user_items):
    """Write user-item interaction file."""
    with open(filepath, 'w') as f:
        for uid in sorted(user_items.keys()):
            items = user_items[uid]
            if items:
                f.write(f"{uid} {' '.join(map(str, items))}\n")


def read_kg(filepath):
    """Read KG triples. Each line: head relation tail"""
    triples = []
    with open(filepath) as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            triples.append((h, r, t))
    return triples


def write_kg(filepath, triples):
    with open(filepath, 'w') as f:
        for h, r, t in triples:
            f.write(f"{h} {r} {t}\n")


def read_mapping(filepath):
    """Read entity/item/user/relation mapping files (with header).
    The remap_id (integer) is always the LAST token on the line for 2-column files,
    and the SECOND token for 3-column files (item_list). org_id can contain spaces.
    """
    mapping = {}
    with open(filepath) as f:
        header = f.readline().strip()
        n_cols = len(header.split())
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)  # split from right: [everything_before, last_token]
            if n_cols == 3:
                # item_list: org_id remap_id freebase_id
                # Split into 3 from the right: last=freebase_id, second-last=remap_id
                tokens = line.rsplit(maxsplit=2)
                if len(tokens) == 3:
                    org_id, remap_id, extra = tokens[0], int(tokens[1]), tokens[2]
                    mapping[remap_id] = (org_id, extra)
                else:
                    continue
            else:
                # 2-column: org_id remap_id — remap_id is last token
                if len(parts) == 2:
                    org_id = parts[0]
                    remap_id = int(parts[1])
                    mapping[remap_id] = (org_id, None)
    return mapping, n_cols


def write_mapping(filepath, entries, header):
    """Write mapping file with header.
    entries: list of tuples to write, one per line, space-separated.
    """
    with open(filepath, 'w') as f:
        f.write(header + '\n')
        for entry in entries:
            f.write(' '.join(str(x) for x in entry) + '\n')


# ──────────────────────────── Core logic ─────────────────────────────

def stratified_user_sample(user_items, ratio, seed=42):
    """
    Sample users via stratified sampling based on interaction count.
    
    WHY THIS PRESERVES CHARACTERISTICS:
    Users are bucketed by degree (number of interactions). We sample
    the same fraction from each bucket. This means:
    - The distribution of user activity levels is preserved
    - Both heavy and light users are represented proportionally
    - The overall sparsity pattern is maintained
    """
    rng = random.Random(seed)
    
    # Bucket users by degree
    degree_buckets = defaultdict(list)
    for uid, items in user_items.items():
        degree = len(items)
        # Logarithmic bucketing to handle long-tail
        bucket = int(np.log2(max(degree, 1)))
        degree_buckets[bucket].append(uid)
    
    sampled_users = set()
    for bucket, users in degree_buckets.items():
        n_sample = max(1, int(len(users) * ratio))
        sampled = rng.sample(users, min(n_sample, len(users)))
        sampled_users.update(sampled)
    
    return sampled_users


def filter_interactions(user_items, sampled_users, retained_items=None):
    """Filter interactions to only include sampled users (and optionally retained items)."""
    filtered = {}
    for uid in sampled_users:
        if uid in user_items:
            items = user_items[uid]
            if retained_items is not None:
                items = [i for i in items if i in retained_items]
            if items:
                filtered[uid] = items
    return filtered


def collect_items(user_items):
    """Collect all unique items from interactions."""
    items = set()
    for uid, item_list in user_items.items():
        items.update(item_list)
    return items


def bfs_kg_subgraph(kg_triples, seed_entities, n_hops=2):
    """
    BFS expansion from seed entities over KG.
    
    WHY THIS PRESERVES CHARACTERISTICS:
    The model (AdaptiveSubgraphModel) does multi-hop message passing
    starting from user/item nodes. By doing K-hop BFS from items,
    we retain exactly the subgraph the model would explore. This means:
    - Local KG structure around items is fully preserved
    - Path patterns used for reasoning are intact
    - Relation type distribution is preserved because we keep all
      edges in the neighborhood (not sampling edges)
    """
    # Build adjacency
    adj = defaultdict(list)  # entity -> [(relation, tail_entity)]
    for h, r, t in kg_triples:
        adj[h].append((r, t))
        adj[t].append((r, h))  # undirected for BFS
    
    visited = set(seed_entities)
    frontier = set(seed_entities)
    retained_triples = set()
    
    for hop in range(n_hops):
        next_frontier = set()
        for entity in frontier:
            for r, neighbor in adj[entity]:
                # Add the triple in its original direction
                retained_triples.add((entity, r, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    
    # Filter to only keep original-direction triples
    original_set = set(kg_triples)
    final_triples = [t for t in retained_triples if t in original_set]
    
    return final_triples, visited


def remap_ids(train_ui, test_ui, kg_triples, item_set):
    """
    Remap all IDs to contiguous ranges starting from 0.
    Returns remapped data and mapping dictionaries.
    """
    # Collect all users
    all_users = sorted(set(train_ui.keys()) | set(test_ui.keys()))
    user_map = {old: new for new, old in enumerate(all_users)}
    
    # Collect all items 
    all_items = sorted(item_set)
    item_map = {old: new for new, old in enumerate(all_items)}
    
    # Collect all KG entities (items + other entities)
    kg_entities = set()
    for h, r, t in kg_triples:
        kg_entities.add(h)
        kg_entities.add(t)
    # Items first, then other entities
    other_entities = sorted(kg_entities - item_set)
    entity_map = {}
    for old_id in all_items:
        entity_map[old_id] = item_map[old_id]
    offset = len(all_items)
    for old_id in other_entities:
        entity_map[old_id] = offset
        offset += 1
    
    # Collect all relations
    all_rels = sorted(set(r for _, r, _ in kg_triples))
    rel_map = {old: new for new, old in enumerate(all_rels)}
    
    # Remap interactions
    new_train = {}
    for uid, items in train_ui.items():
        new_uid = user_map[uid]
        new_items = [item_map[i] for i in items if i in item_map]
        if new_items:
            new_train[new_uid] = new_items
    
    new_test = {}
    for uid, items in test_ui.items():
        new_uid = user_map[uid]
        new_items = [item_map[i] for i in items if i in item_map]
        if new_items:
            new_test[new_uid] = new_items
    
    # Remap KG
    new_kg = []
    for h, r, t in kg_triples:
        if h in entity_map and t in entity_map and r in rel_map:
            new_kg.append((entity_map[h], rel_map[r], entity_map[t]))
    
    return new_train, new_test, new_kg, user_map, item_map, entity_map, rel_map


# ─────────────────────── Validation / Evidence ───────────────────────

def compute_statistics(train_ui, test_ui, kg_triples, label=""):
    """Compute comprehensive statistics for a dataset."""
    stats = {}
    
    # -- Interaction stats --
    all_users = set(train_ui.keys()) | set(test_ui.keys())
    train_items = collect_items(train_ui)
    test_items = collect_items(test_ui)
    all_items = train_items | test_items
    
    train_interactions = sum(len(v) for v in train_ui.values())
    test_interactions = sum(len(v) for v in test_ui.values())
    total_interactions = train_interactions + test_interactions
    
    stats['n_users'] = len(all_users)
    stats['n_items'] = len(all_items)
    stats['n_train_interactions'] = train_interactions
    stats['n_test_interactions'] = test_interactions
    stats['train_test_ratio'] = train_interactions / max(test_interactions, 1)
    stats['sparsity'] = 1.0 - total_interactions / (len(all_users) * max(len(all_items), 1))
    
    # User degree distribution
    user_degrees = []
    for uid in all_users:
        deg = len(train_ui.get(uid, [])) + len(test_ui.get(uid, []))
        user_degrees.append(deg)
    user_degrees = np.array(user_degrees)
    stats['user_deg_mean'] = np.mean(user_degrees)
    stats['user_deg_median'] = np.median(user_degrees)
    stats['user_deg_std'] = np.std(user_degrees)
    stats['user_deg_p25'] = np.percentile(user_degrees, 25)
    stats['user_deg_p75'] = np.percentile(user_degrees, 75)
    stats['user_deg_p90'] = np.percentile(user_degrees, 90)
    
    # Item degree (popularity) distribution
    item_counter = Counter()
    for items in train_ui.values():
        item_counter.update(items)
    for items in test_ui.values():
        item_counter.update(items)
    item_degrees = np.array(sorted(item_counter.values())) if item_counter else np.array([0])
    stats['item_deg_mean'] = np.mean(item_degrees)
    stats['item_deg_median'] = np.median(item_degrees)
    stats['item_deg_std'] = np.std(item_degrees)
    # Gini coefficient: measures inequality of item popularity (long-tail shape)
    # Preserved Gini = preserved popularity distribution shape
    n = len(item_degrees)
    if n > 0 and np.sum(item_degrees) > 0:
        index = np.arange(1, n + 1)
        stats['item_gini'] = (2 * np.sum(index * item_degrees) / (n * np.sum(item_degrees))) - (n + 1) / n
    else:
        stats['item_gini'] = 0.0
    # Coefficient of variation: shape-invariant measure
    stats['item_cv'] = np.std(item_degrees) / max(np.mean(item_degrees), 1e-10)
    
    # -- KG stats --
    kg_entities = set()
    kg_relations = set()
    for h, r, t in kg_triples:
        kg_entities.add(h)
        kg_entities.add(t)
        kg_relations.add(r)
    
    stats['n_kg_entities'] = len(kg_entities)
    stats['n_kg_relations'] = len(kg_relations)
    stats['n_kg_triples'] = len(kg_triples)
    
    # KG entity degree distribution
    kg_deg = Counter()
    for h, r, t in kg_triples:
        kg_deg[h] += 1
        kg_deg[t] += 1
    kg_degrees = np.array(list(kg_deg.values())) if kg_deg else np.array([0])
    stats['kg_deg_mean'] = np.mean(kg_degrees)
    stats['kg_deg_median'] = np.median(kg_degrees)
    stats['kg_deg_std'] = np.std(kg_degrees)
    
    # Relation type distribution (entropy as proxy)
    rel_counts = Counter(r for _, r, _ in kg_triples)
    total_triples = len(kg_triples) if kg_triples else 1
    rel_probs = np.array([c / total_triples for c in rel_counts.values()])
    stats['rel_entropy'] = -np.sum(rel_probs * np.log(rel_probs + 1e-10))
    
    # Items in KG coverage
    items_in_kg = len(all_items & kg_entities)
    stats['item_kg_coverage'] = items_in_kg / max(len(all_items), 1)
    
    # Test user coverage in train
    test_users_in_train = len(set(test_ui.keys()) & set(train_ui.keys()))
    stats['test_user_coverage'] = test_users_in_train / max(len(test_ui), 1)
    
    return stats


def print_comparison(orig_stats, sub_stats, dataset_name):
    """Print side-by-side comparison with preservation ratios."""
    print("\n" + "=" * 90)
    print(f"  EVIDENCE REPORT: {dataset_name}")
    print("=" * 90)
    print(f"{'Metric':<35} {'Original':>15} {'Subset':>15} {'Ratio':>10} {'Preserved?':>12}")
    print("-" * 90)
    
    metrics = [
        ('n_users',                'Users',                   None),
        ('n_items',                'Items',                   None),
        ('n_train_interactions',   'Train interactions',      None),
        ('n_test_interactions',    'Test interactions',       None),
        ('train_test_ratio',       'Train/Test ratio',        0.15),
        ('sparsity',               'Sparsity',                0.05),
        ('user_deg_mean',          'User degree mean',        0.20),
        ('user_deg_median',        'User degree median',      0.20),
        ('user_deg_std',           'User degree std',         0.30),
        ('user_deg_p25',           'User degree P25',         0.25),
        ('user_deg_p75',           'User degree P75',         0.25),
        ('user_deg_p90',           'User degree P90',         0.25),
        ('item_deg_mean',          'Item popularity mean',    None),
        ('item_deg_median',        'Item popularity median',  None),
        ('item_gini',              'Item Gini (shape)',       0.15),
        ('item_cv',                'Item CV (shape)',         0.25),
        ('n_kg_entities',          'KG entities',             None),
        ('n_kg_relations',         'KG relations',            0.10),
        ('n_kg_triples',           'KG triples',              None),
        ('kg_deg_mean',            'KG degree mean',          0.25),
        ('kg_deg_median',          'KG degree median',        0.30),
        ('rel_entropy',            'Relation entropy',        0.10),
        ('item_kg_coverage',       'Item-KG coverage',        0.10),
        ('test_user_coverage',     'Test user in train',      0.05),
    ]
    
    all_good = True
    for key, label, threshold in metrics:
        o = orig_stats[key]
        s = sub_stats[key]
        
        if threshold is not None:
            # Check relative difference
            if o != 0:
                rel_diff = abs(s - o) / abs(o)
            else:
                rel_diff = 0 if s == 0 else 1
            preserved = "YES" if rel_diff <= threshold else f"NO ({rel_diff:.1%})"
            if rel_diff > threshold:
                all_good = False
        else:
            preserved = "---"  # Size metrics, just show reduction
        
        # Format based on type
        if isinstance(o, float):
            print(f"{label:<35} {o:>15.4f} {s:>15.4f} {s/max(o,1e-10):>9.2%} {preserved:>12}")
        else:
            print(f"{label:<35} {o:>15,} {s:>15,} {s/max(o,1):>9.2%} {preserved:>12}")
    
    print("-" * 90)
    if all_good:
        print("  VERDICT: All distributional characteristics are well-preserved.")
    else:
        print("  VERDICT: Some metrics deviate. Consider adjusting ratio or n_hops.")
    print("=" * 90)


# ──────────────────────────── Main pipeline ──────────────────────────

def subset_standard_dataset(data_dir, output_dir, ratio, n_hops, seed):
    """Subset datasets with standard format (train.txt, test.txt)."""
    print(f"\n>>> Processing {data_dir} (standard format)")
    
    train_ui = read_cf(os.path.join(data_dir, 'train.txt'))
    test_ui = read_cf(os.path.join(data_dir, 'test.txt'))
    kg_triples = read_kg(os.path.join(data_dir, 'kg.txt'))
    
    # Merge for user degree computation
    all_ui = defaultdict(list)
    for uid, items in train_ui.items():
        all_ui[uid].extend(items)
    for uid, items in test_ui.items():
        all_ui[uid].extend(items)
    
    # Step 1: Stratified user sampling
    sampled_users = stratified_user_sample(all_ui, ratio, seed)
    print(f"  Sampled {len(sampled_users)} / {len(all_ui)} users ({len(sampled_users)/len(all_ui):.1%})")
    
    # Step 2: Filter interactions & collect items
    sub_train = filter_interactions(train_ui, sampled_users)
    sub_test = filter_interactions(test_ui, sampled_users)
    
    # Ensure test users appear in train
    sub_test = {u: items for u, items in sub_test.items() if u in sub_train}
    
    item_set = collect_items(sub_train) | collect_items(sub_test)
    print(f"  Retained {len(item_set)} items")
    
    # Step 3: BFS KG expansion from items
    sub_kg, kg_entities = bfs_kg_subgraph(kg_triples, item_set, n_hops)
    print(f"  KG: {len(sub_kg)} triples, {len(kg_entities)} entities (from {len(kg_triples)} triples)")
    
    # Step 4: Remap IDs
    new_train, new_test, new_kg, user_map, item_map, entity_map, rel_map = \
        remap_ids(sub_train, sub_test, sub_kg, item_set)
    
    # Step 5: Write output
    os.makedirs(output_dir, exist_ok=True)
    write_cf(os.path.join(output_dir, 'train.txt'), new_train)
    write_cf(os.path.join(output_dir, 'test.txt'), new_test)
    write_kg(os.path.join(output_dir, 'kg.txt'), new_kg)
    
    # Write mapping files
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}
    inv_entity_map = {v: k for k, v in entity_map.items()}
    inv_rel_map = {v: k for k, v in rel_map.items()}
    
    # user_list.txt
    orig_user_map = {}
    user_list_path = os.path.join(data_dir, 'user_list.txt')
    if os.path.exists(user_list_path):
        orig_user_map, _ = read_mapping(user_list_path)
    entries = []
    for new_id in sorted(inv_user_map.keys()):
        old_id = inv_user_map[new_id]
        org_id = orig_user_map.get(old_id, (str(old_id), None))[0] if orig_user_map else str(old_id)
        entries.append((org_id, new_id))
    write_mapping(os.path.join(output_dir, 'user_list.txt'), entries, 'org_id remap_id')
    
    # item_list.txt
    orig_item_map = {}
    item_list_path = os.path.join(data_dir, 'item_list.txt')
    if os.path.exists(item_list_path):
        orig_item_map, n_cols = read_mapping(item_list_path)
    entries = []
    for new_id in sorted(inv_item_map.keys()):
        old_id = inv_item_map[new_id]
        if orig_item_map and old_id in orig_item_map:
            org_id, extra = orig_item_map[old_id]
            if extra:
                entries.append((org_id, new_id, extra))
            else:
                entries.append((org_id, new_id))
        else:
            entries.append((str(old_id), new_id))
    header = 'org_id remap_id freebase_id' if (orig_item_map and any(v[1] for v in orig_item_map.values())) else 'org_id remap_id'
    write_mapping(os.path.join(output_dir, 'item_list.txt'), entries, header)
    
    # entity_list.txt
    orig_entity_map = {}
    entity_list_path = os.path.join(data_dir, 'entity_list.txt')
    if os.path.exists(entity_list_path):
        orig_entity_map, _ = read_mapping(entity_list_path)
    entries = []
    for new_id in sorted(inv_entity_map.keys()):
        old_id = inv_entity_map[new_id]
        org_id = orig_entity_map.get(old_id, (str(old_id), None))[0] if orig_entity_map else str(old_id)
        entries.append((org_id, new_id))
    write_mapping(os.path.join(output_dir, 'entity_list.txt'), entries, 'org_id remap_id')
    
    # relation_list.txt
    orig_rel_map = {}
    rel_list_path = os.path.join(data_dir, 'relation_list.txt')
    if os.path.exists(rel_list_path):
        orig_rel_map, _ = read_mapping(rel_list_path)
    entries = []
    for new_id in sorted(inv_rel_map.keys()):
        old_id = inv_rel_map[new_id]
        org_id = orig_rel_map.get(old_id, (str(old_id), None))[0] if orig_rel_map else str(old_id)
        entries.append((org_id, new_id))
    write_mapping(os.path.join(output_dir, 'relation_list.txt'), entries, 'org_id remap_id')
    
    # Step 6: Compute stats and print evidence
    orig_stats = compute_statistics(train_ui, test_ui, kg_triples, "original")
    sub_stats = compute_statistics(new_train, new_test, new_kg, "subset")
    dataset_name = Path(data_dir).name
    print_comparison(orig_stats, sub_stats, dataset_name)
    
    return orig_stats, sub_stats


def subset_new_dataset(data_dir, output_dir, ratio, n_hops, seed):
    """Subset datasets with new format (train_1.txt, test_1.txt)."""
    print(f"\n>>> Processing {data_dir} (new format)")
    
    train_ui = read_cf(os.path.join(data_dir, 'train_1.txt'))
    test_ui = read_cf(os.path.join(data_dir, 'test_1.txt'))
    kg_triples = read_kg(os.path.join(data_dir, 'kg.txt'))
    
    all_ui = defaultdict(list)
    for uid, items in train_ui.items():
        all_ui[uid].extend(items)
    for uid, items in test_ui.items():
        all_ui[uid].extend(items)
    
    # Step 1: Stratified user sampling
    sampled_users = stratified_user_sample(all_ui, ratio, seed)
    print(f"  Sampled {len(sampled_users)} / {len(all_ui)} users ({len(sampled_users)/len(all_ui):.1%})")
    
    # Step 2: Filter interactions
    sub_train = filter_interactions(train_ui, sampled_users)
    sub_test = filter_interactions(test_ui, sampled_users)
    sub_test = {u: items for u, items in sub_test.items() if u in sub_train}
    
    item_set = collect_items(sub_train) | collect_items(sub_test)
    print(f"  Retained {len(item_set)} items")
    
    # Step 3: BFS KG expansion
    sub_kg, kg_entities = bfs_kg_subgraph(kg_triples, item_set, n_hops)
    print(f"  KG: {len(sub_kg)} triples, {len(kg_entities)} entities")
    
    # Step 4: Remap
    new_train, new_test, new_kg, user_map, item_map, entity_map, rel_map = \
        remap_ids(sub_train, sub_test, sub_kg, item_set)
    
    # Step 5: Write
    os.makedirs(output_dir, exist_ok=True)
    write_cf(os.path.join(output_dir, 'train_1.txt'), new_train)
    write_cf(os.path.join(output_dir, 'test_1.txt'), new_test)
    write_kg(os.path.join(output_dir, 'kg.txt'), new_kg)
    
    # Step 6: Evidence
    orig_stats = compute_statistics(train_ui, test_ui, kg_triples)
    sub_stats = compute_statistics(new_train, new_test, new_kg)
    dataset_name = Path(data_dir).name
    print_comparison(orig_stats, sub_stats, dataset_name)
    
    return orig_stats, sub_stats


def main():
    parser = argparse.ArgumentParser(
        description='Create subsets of KGRS datasets preserving statistical characteristics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Subset all datasets to 10%
  python subset_data.py --ratio 0.1

  # Subset only last-fm to 20% with 3-hop KG
  python subset_data.py --datasets last-fm --ratio 0.2 --n_hops 3

  # Custom output suffix
  python subset_data.py --ratio 0.1 --suffix _small
        """
    )
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='Fraction of users to sample (default: 0.1 = 10%%)')
    parser.add_argument('--n_hops', type=int, default=2,
                        help='Number of BFS hops for KG expansion (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--suffix', type=str, default='_subset',
                        help='Suffix for output folder names (default: _subset)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to process. Default: all 6 datasets')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root data directory (default: same directory as this script)')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = os.path.dirname(os.path.abspath(__file__))
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Define all datasets
    standard_datasets = ['alibaba-fashion', 'amazon-book', 'last-fm', 'yelp2018', 'yelp2018-kgcl']
    new_datasets = ['new_alibaba-fashion', 'new_amazon-book', 'new_last-fm']
    
    if args.datasets:
        selected = set(args.datasets)
        standard_datasets = [d for d in standard_datasets if d in selected]
        new_datasets = [d for d in new_datasets if d in selected]
    
    all_results = {}
    
    for dataset in standard_datasets:
        data_dir = os.path.join(args.data_root, dataset)
        output_dir = os.path.join(args.data_root, dataset + args.suffix)
        if os.path.isdir(data_dir):
            orig, sub = subset_standard_dataset(data_dir, output_dir, args.ratio, args.n_hops, args.seed)
            all_results[dataset] = (orig, sub)
        else:
            print(f"  WARNING: {data_dir} not found, skipping.")
    
    for dataset in new_datasets:
        data_dir = os.path.join(args.data_root, dataset)
        output_dir = os.path.join(args.data_root, dataset + args.suffix)
        if os.path.isdir(data_dir):
            orig, sub = subset_new_dataset(data_dir, output_dir, args.ratio, args.n_hops, args.seed)
            all_results[dataset] = (orig, sub)
        else:
            print(f"  WARNING: {data_dir} not found, skipping.")
    
    # Summary
    print("\n\n" + "=" * 90)
    print("  SUMMARY: SIZE REDUCTION ACROSS ALL DATASETS")
    print("=" * 90)
    print(f"{'Dataset':<25} {'Users':>10} {'Items':>10} {'Interactions':>15} {'KG Triples':>12}")
    print("-" * 90)
    for name, (orig, sub) in all_results.items():
        u_pct = sub['n_users'] / orig['n_users'] * 100
        i_pct = sub['n_items'] / orig['n_items'] * 100
        inter_pct = (sub['n_train_interactions'] + sub['n_test_interactions']) / \
                    (orig['n_train_interactions'] + orig['n_test_interactions']) * 100
        kg_pct = sub['n_kg_triples'] / max(orig['n_kg_triples'], 1) * 100
        print(f"{name:<25} {u_pct:>9.1f}% {i_pct:>9.1f}% {inter_pct:>14.1f}% {kg_pct:>11.1f}%")
    print("=" * 90)
    print(f"\nSubset datasets written with suffix '{args.suffix}'")
    print("You can now use them with the DataLoader by pointing task_dir to the new folders.")


if __name__ == '__main__':
    main()

# example usage
# # Subset all 6 datasets to 10%
# python data/subset_data.py --ratio 0.1

# # Subset specific dataset to 20% with 3-hop KG
# python data/subset_data.py --datasets last-fm --ratio 0.2 --n_hops 3

# # Then use in training
# python train.py --task_dir data/last-fm_subset/ ...