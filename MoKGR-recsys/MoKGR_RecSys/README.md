# MoKGR for Recommendation

Adaptation of MoKGR (Mixture of Experts for Knowledge Graph Reasoning) from KG completion to recommendation tasks.

## What Is MoKGR

MoKGR is a GNN-based reasoning model with two key innovations:
1. **MoE for Hops** — Learns which propagation depths (hop distances) are most relevant per query via a mixture-of-experts that selects top-K hops adaptively.
2. **MoE for Pruning** — Uses three complementary pruning experts (node-score, similarity, attention-based) to dynamically filter the explored subgraph at each layer.

Additionally, it uses a **Gumbel-Sigmoid gate** for early stopping and **GRU gating** across layers.

## Adaptation from KG Completion to Recommendation

### Key Changes

| Aspect | KG Completion (Original) | Recommendation (This) |
|--------|--------------------------|----------------------|
| **Task** | Predict missing entity in `(h, r, ?)` | Predict items a user will interact with |
| **Graph** | KG entities only `[0, n_ent)` | Users `[0, n_users)` + KG entities `[n_users, n_nodes)` |
| **Relations** | `2*n_rel + 1` (original + inverse + self-loop) | `2*n_rel + 3` (+ interact, inv-interact) |
| **Loss** | Ranking loss (softmax CE) | BPR loss + MoE regularization losses |
| **Output** | Scores for all entities `(batch, n_ent)` | Scores for items only `(batch, n_items)` |
| **Metrics** | MRR, Hit@1, Hit@10 | Recall@20, NDCG@20 |
| **MoE Pruning** | Operates on `n_ent` nodes | Operates on `n_nodes` (users + entities) |

### What Was Preserved

- **MoE for Hops**: Adaptive path length selection with learned hop embeddings, importance/load balancing losses
- **MoE for Pruning**: Three complementary experts (nodes pruner, similarity pruner, alpha pruner) with dynamic top-K per layer
- **Gumbel-Sigmoid Gate**: Early stopping mechanism for graph exploration
- **GRU Gating**: Layer-wise hidden state combination
- **PPR Sampling**: Optional Personalized PageRank subgraph construction
- All MoE regularization losses (L_importance, L_load, L_importance_pruning) — these are architectural, not task-specific

### Training Loss

```
total_loss = BPR_loss
           + λ_importance × L_importance      (MoE hops: expert balance)
           + λ_load × L_load                  (MoE hops: load balance)
           + λ_pruning × L_importance_pruning  (MoE pruning: expert balance)
```

### Relation Mapping

```
0                        : interact     (user → item)
1                        : inv-interact (item → user)
2 ... n_rel+1            : original KG relations (shifted +2)
n_rel+2 ... 2*n_rel+1    : inverse KG relations
2*n_rel+2                : self-loop
```

### Entity Mapping

```
[0, n_users)                    : user nodes
[n_users, n_users + n_items)    : item nodes (subset of KG entities)
[n_users, n_users + n_ent)      : all KG entities (items + non-item entities)
```

### Score Accumulation

At each hop within `[min_hop, max_hop]`:
1. Compute raw scores via `W_final(hidden)`
2. Apply MoE pruning (filter nodes)
3. Multiply by MoE hop weight `G_full[j]`
4. **Filter to item nodes only** (`node_id >= n_users` and `< n_users + n_items`)
5. Accumulate into `scores_all` of shape `(batch, n_items)`

## File Structure

```
MoKGR_RecSys/
├── train.py          # Entry point, per-dataset hyperparameter presets
├── base_model.py     # Training loop (BPR + MoE losses), evaluation (Recall@K, NDCG@K)
├── models.py         # MoKGR_RecSys: GNNLayer + MoE hops + MoE pruning + gate + item readout
├── moe.py            # MoE_for_hops and MoE_for_Pruning modules
├── load_data.py      # Data loader: user-item CF + KG → unified graph, negative sampling
├── utils.py          # BPR loss, NDCG/DCG computation
├── PPR_sampler.py    # Personalized PageRank subgraph sampler (optional, needs CuPy)
└── data/             # Symlink to datasets (last-fm, amazon-book, alibaba-fashion, ...)
```

### File Details

- **`train.py`** — CLI entry point. Per-dataset hyperparameter presets for all rec datasets. Key MoKGR args: `--max_hop`, `--min_hop`, `--num_experts`, `--num_pruning_experts`, `--active_gate`, `--active_PPR`.

- **`models.py`** — `MoKGR_RecSys` model. Stacks `GNNLayer` modules with MoE for hops (adaptive depth), MoE for pruning (complementary experts), Gumbel-Sigmoid gate (early stopping), and GRU gating. Score accumulation filters to item nodes only.

- **`moe.py`** — Two MoE modules:
  - `MoE_for_hops`: Selects top-K hop distances via learned hop embeddings + context MLP. Includes importance and load balancing losses.
  - `MoE_for_Pruning`: Three pruning experts (node-score, cosine-similarity, attention-based). Dynamic K per layer via sigmoid S-curve. Expert gating via learned weights.

- **`base_model.py`** — Training harness. `train_batch()` uses BPR loss + MoE regularization. `test_batch()` evaluates Recall@20 and NDCG@20.

- **`load_data.py`** — Builds unified user+entity graph. Reads `train.txt`/`test.txt` (user-item) and `kg.txt` (KG triples). Optional PPR subgraph sampling via `--active_PPR`.

- **`PPR_sampler.py`** — GPU-accelerated Personalized PageRank using CuPy. Computes PPR scores for subgraph construction. Only needed if `--active_PPR` is used.

- **`utils.py`** — `cal_bpr_loss()` for training. `ndcg_k()` and `dcg_k()` for evaluation.

## Dependencies

```
torch>=2.0          # with CUDA
torch_scatter        # must match torch+CUDA version
scipy
numpy
tqdm
cupy-cuda12x         # optional, only for --active_PPR
```

### Install (torch 2.8.0 + CUDA 12.8)

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install scipy numpy tqdm

# Optional: for PPR sampling
pip install cupy-cuda12x
```

## Usage

### Training

```bash
# last-fm (default)
python train.py --data_path data/last-fm/ --max_hop 4 --min_hop 2 --gpu 0

# amazon-book
python train.py --data_path data/amazon-book/ --max_hop 5 --min_hop 2 --gpu 0

# alibaba-fashion
python train.py --data_path data/alibaba-fashion/ --max_hop 5 --min_hop 2 --gpu 0

# With PPR subgraph sampling (requires cupy)
python train.py --data_path data/last-fm/ --max_hop 4 --min_hop 2 --active_PPR --gpu 0

# With Gumbel-Sigmoid gate
python train.py --data_path data/last-fm/ --max_hop 6 --min_hop 2 --active_gate --gpu 0
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/last-fm/` | Path to dataset |
| `--max_hop` | 5 | Maximum propagation depth (also sets n_layer) |
| `--min_hop` | 2 | Minimum hop for MoE scoring |
| `--num_experts` | 3 | Top-K hops to select (MoE for hops) |
| `--num_pruning_experts` | 2 | Number of pruning experts to activate |
| `--active_gate` | off | Enable Gumbel-Sigmoid early stopping gate |
| `--active_PPR` | off | Enable PPR subgraph sampling |
| `--K_source` | 1000 | Initial pruning budget |
| `--K_max` | 2000 | Peak pruning budget |
| `--K_min` | 1000 | Final pruning budget |
| `--epoch` | 40 | Training epochs |
| `--gpu` | 0 | GPU device ID |

## Data Format

Each dataset folder should contain:

```
dataset/
├── train.txt       # user_id item1 item2 item3 ...  (one line per user)
├── test.txt        # same format
└── kg.txt          # head_entity_id relation_id tail_entity_id  (one triple per line)
```

## Evaluation

After each training epoch, the model evaluates on the test set:
- **Recall@20**: Fraction of ground-truth items appearing in top-20 predictions
- **NDCG@20**: Position-aware ranking quality

Results are logged to `results/<dataset>_perf.txt`.

## Compared to Original MoKGR (KG Completion)

The original transductive MoKGR is in `../transductive/`. It uses:
- KG triple format (`entities.txt`, `relations.txt`, `facts.txt`, `train.txt`, `valid.txt`, `test.txt`)
- Ranking loss (softmax cross-entropy)
- MRR / Hit@1 / Hit@10 metrics
- Entity-level scoring

This RecSys version preserves all core MoKGR mechanisms while adapting the task interface.
