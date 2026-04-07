from tqdm import tqdm
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


class PPRSampler:
    def __init__(self, n_ent, edges, sampling_percentage=0.8, PPR_alpha=0.85, max_iter=100, tol=1e-9):
        print('==> Initializing PPRSampler...')
        self.n_ent = n_ent
        self.edges = edges
        self.sampling_percentage = sampling_percentage
        self.PPR_alpha = PPR_alpha
        self.max_iter = max_iter
        self.tol = tol

        self.adjacency_matrix = self._build_adjacency_matrix()
        print('==> Initialization completed.')

    def _build_adjacency_matrix(self):
        src, dst = zip(*self.edges)
        src = cp.array(src, dtype=cp.int32)
        dst = cp.array(dst, dtype=cp.int32)
        data = cp.ones(len(src), dtype=cp.float32)

        adjacency_matrix = cpx_sparse.coo_matrix((data, (src, dst)), shape=(self.n_ent, self.n_ent))

        # Undirected graph
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T

        # Convert to CSR for efficient arithmetic
        adjacency_matrix = adjacency_matrix.tocsr()

        # Row-normalize
        row_sums = cp.array(adjacency_matrix.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_row_sums = cpx_sparse.diags(1.0 / row_sums)
        adjacency_matrix = inv_row_sums @ adjacency_matrix

        return adjacency_matrix

    def compute_ppr_for_seed(self, seed):
        teleport = cp.zeros(self.n_ent, dtype=cp.float32)
        teleport[seed] = 1.0

        pr = cp.ones(self.n_ent, dtype=cp.float32) / self.n_ent

        for i in range(self.max_iter):
            new_pr = self.PPR_alpha * self.adjacency_matrix.dot(pr) + (1 - self.PPR_alpha) * teleport
            if cp.linalg.norm(new_pr - pr, ord=1) < self.tol:
                pr = new_pr
                break
            pr = new_pr

        return cp.asnumpy(pr)

    def compute_ppr_batch(self, seeds):
        """Compute PPR for a batch of seeds in parallel on GPU.

        Instead of one power iteration per seed, runs all seeds as columns
        of a matrix so each iteration is a single sparse matmul.

        Parameters
        ----------
        seeds : list[int]
            Batch of seed node IDs.

        Returns
        -------
        dict[int, ndarray]
            Mapping from seed to PPR score array (numpy, shape [n_ent]).
        """
        batch_size = len(seeds)

        # Teleport matrix: each column is a one-hot for a seed
        teleport = cp.zeros((self.n_ent, batch_size), dtype=cp.float32)
        for i, seed in enumerate(seeds):
            teleport[seed, i] = 1.0

        # PR matrix: columns are PPR vectors being iterated
        pr = cp.ones((self.n_ent, batch_size), dtype=cp.float32) / self.n_ent

        for _ in range(self.max_iter):
            # A @ PR is sparse @ dense, CuPy handles this efficiently
            new_pr = self.PPR_alpha * (self.adjacency_matrix @ pr) + (1 - self.PPR_alpha) * teleport
            # Check convergence across all columns
            diff = cp.linalg.norm(new_pr - pr, ord=1, axis=0).max()
            pr = new_pr
            if diff < self.tol:
                break

        # Transfer to CPU
        pr_cpu = cp.asnumpy(pr)

        return {seed: pr_cpu[:, i] for i, seed in enumerate(seeds)}

    def sample_nodes(self, seeds, batch_size=64):
        """Compute PPR for all seeds, processing in batches on GPU.

        Parameters
        ----------
        seeds : list[int]
            All seed node IDs to compute PPR for.
        batch_size : int
            Number of seeds to process in parallel per GPU batch.
            Higher = faster but uses more GPU memory.
            Rule of thumb: ~64 for 8GB VRAM, ~256 for 24GB+ VRAM.
        """
        ppr_scores_per_seed = {}
        n_seeds = len(seeds)
        n_batches = (n_seeds + batch_size - 1) // batch_size

        print(f'==> Calculating PPR scores on GPU (batch_size={batch_size})...')
        for b in tqdm(range(n_batches), desc='PPR batches'):
            start = b * batch_size
            end = min(n_seeds, (b + 1) * batch_size)
            batch_seeds = seeds[start:end]
            batch_results = self.compute_ppr_batch(batch_seeds)
            ppr_scores_per_seed.update(batch_results)

        return ppr_scores_per_seed
