import numpy as np
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA


class FastBlockPCA:
    """Block-wise PCA used by ACFDE for fast dimensionality reduction."""

    def __init__(self, ndim: int, n_blocks: int, dtype: np.dtype = np.float32):
        if ndim <= 0 or n_blocks <= 0:
            raise ValueError("ndim and n_blocks must be positive")

        self.ndim = int(ndim)
        self.n_blocks = int(n_blocks)
        self.dtype = dtype

        self.block_slices: List[slice] = self._make_block_slices(self.ndim, self.n_blocks)
        self.P_blocks: List[np.ndarray] = []
        self.lmd_blocks: List[np.ndarray] = []
        self.flat2block_local: List[Tuple[int, int]] = []

        self.keep_dim: Optional[int] = None
        self.global_topk_flat: Optional[np.ndarray] = None
        self.global_topk_lmd: Optional[np.ndarray] = None

        self.kept_local_dims: List[np.ndarray] = []
        self.kept_outpos: List[np.ndarray] = []
        self.W_global: Optional[np.ndarray] = None
        self._fitted = False

    @staticmethod
    def _make_block_slices(ndim: int, n_blocks: int) -> List[slice]:
        base = ndim // n_blocks
        rem = ndim % n_blocks
        out: List[slice] = []
        start = 0
        for b in range(n_blocks):
            width = base + (1 if b < rem else 0)
            end = start + width
            out.append(slice(start, end))
            start = end
        return out

    @staticmethod
    def _fit_pca(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=None)
        pca.fit(x)
        w = pca.components_.T.astype(np.float32)
        lmd = pca.explained_variance_.astype(np.float32)
        return w, lmd

    def fit(
        self,
        data: np.ndarray,
        sample_ratio: float = 1.0,
        keep_dim: Optional[int] = None,
        seed: int = 42,
    ) -> "FastBlockPCA":
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != self.ndim:
            raise ValueError(f"Expect (n, {self.ndim}), got {data.shape}")

        n_docs = data.shape[0]
        if sample_ratio < 1.0:
            rng = np.random.default_rng(seed)
            sample_size = max(1, int(n_docs * sample_ratio))
            idx = rng.choice(n_docs, size=sample_size, replace=False)
            data_fit = data[idx]
        else:
            data_fit = data

        self.P_blocks = []
        self.lmd_blocks = []
        self.flat2block_local = []

        lmd_all = []
        for b, sl in enumerate(self.block_slices):
            xb = data_fit[:, sl].astype(self.dtype, copy=False)
            if xb.shape[0] <= xb.shape[1]:
                raise ValueError(
                    f"Block {b} has #samples <= #dims ({xb.shape[0]} <= {xb.shape[1]}). "
                    "Increase sample_ratio or reduce projection dimensions."
                )

            p, lmd = self._fit_pca(xb)
            self.P_blocks.append(p.astype(self.dtype, copy=False))
            self.lmd_blocks.append(lmd.astype(self.dtype, copy=False))

            for j in range(lmd.shape[0]):
                self.flat2block_local.append((b, j))
            lmd_all.append(lmd)

        lmd_all = np.concatenate(lmd_all, axis=0)
        total_dims = int(lmd_all.shape[0])
        k = total_dims if keep_dim is None else min(int(keep_dim), total_dims)
        self.keep_dim = k

        if k >= total_dims:
            order = np.argsort(lmd_all)[::-1]
        else:
            kth = total_dims - k
            idx_part = np.argpartition(lmd_all, kth)[kth:]
            order = idx_part[np.argsort(lmd_all[idx_part])[::-1]]

        self.global_topk_flat = order.astype(np.int64, copy=False)
        self.global_topk_lmd = lmd_all[self.global_topk_flat].astype(self.dtype, copy=False)

        kept_local: List[List[int]] = [[] for _ in range(self.n_blocks)]
        kept_outpos: List[List[int]] = [[] for _ in range(self.n_blocks)]

        for out_pos, flat_idx in enumerate(self.global_topk_flat):
            b, j = self.flat2block_local[int(flat_idx)]
            kept_local[b].append(j)
            kept_outpos[b].append(out_pos)

        self.kept_local_dims = [np.asarray(v, dtype=np.int64) for v in kept_local]
        self.kept_outpos = [np.asarray(v, dtype=np.int64) for v in kept_outpos]

        w = np.zeros((self.ndim, int(self.keep_dim)), dtype=self.dtype, order="F")
        for b, sl in enumerate(self.block_slices):
            locs = self.kept_local_dims[b]
            outs = self.kept_outpos[b]
            if locs.size == 0:
                continue
            w[sl, outs] = self.P_blocks[b][:, locs]

        self.W_global = w
        self._fitted = True
        return self

    def transform(self, data: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted or self.W_global is None or self.keep_dim is None:
            raise RuntimeError("FastBlockPCA is not fitted. Call fit() first.")

        x = np.asarray(data)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(f"Expect (n, {self.ndim}), got {x.shape}")

        if x.dtype != self.dtype:
            x = x.astype(self.dtype, copy=False)
        if not (x.flags.c_contiguous or x.flags.f_contiguous):
            x = np.ascontiguousarray(x)

        n = x.shape[0]
        k = int(self.keep_dim)

        if out is None:
            out = np.empty((n, k), dtype=self.dtype)
        else:
            if out.shape != (n, k) or out.dtype != self.dtype:
                raise ValueError(f"out must have shape {(n, k)} and dtype {self.dtype}")

        np.matmul(x, self.W_global, out=out)
        return out
