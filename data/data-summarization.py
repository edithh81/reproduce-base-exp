import numpy as np

def read_cf(file):
    inter_mat = []
    with open(file, 'r') as f:
        for line in f:
            arr = list(map(int, line.strip().split()))
            u, items = arr[0], arr[1:]
            for i in items: # Allow duplicate interactions
                inter_mat.append((u, i))
    return np.array(inter_mat)

def read_triples(file):
    triples = []
    with open(file, 'r') as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            triples.append((h, r, t))
    return np.array(triples)

def summarize(data_dir):
    train_cf = read_cf(f"{data_dir}/train.txt")
    test_cf = read_cf(f"{data_dir}/test.txt")
    all_cf = np.concatenate([train_cf, test_cf], axis=0)
    triples = read_triples(f"{data_dir}/kg.txt")

    num_users = len(set(all_cf[:,0]))
    num_items = len(set(all_cf[:,1]))
    num_inter = all_cf.shape[0]
    num_entities = len(
        (
            set(triples[:,0]).union(set(triples[:,2]))
        ).difference(set(all_cf[:,1]))
    )
    num_relations = len(set(triples[:,1]))
    num_triplets = triples.shape[0]

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_inter}")
    print(f"Entities: {num_entities}")
    print(f"Relations: {num_relations}")
    print(f"Triplets: {num_triplets}")

# Example usage:
summarize('data/last-fm-lightkg')