import os, time
import psutil
from contextlib import contextmanager
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Union
from numba import njit, prange

_PROC = psutil.Process(os.getpid())
_MB = 1024 ** 2

def _rss_mb() -> float:
    return _PROC.memory_info().rss / _MB

class TrackStep:
    def __init__(self, step: str):
        self.step = step
        self.dt = None
        self.rss0 = None
        self.rss1 = None
        self.drss = None

    def __enter__(self):
        self.rss0 = _rss_mb()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
        self.rss1 = _rss_mb()
        self.drss = self.rss1 - self.rss0
        print(
            f"[{self.step}] time={self.dt:.3f}s | "
            f"RSS {self.rss0:.2f} -> {self.rss1:.2f} MB | ΔRSS {self.drss:+.2f} MB"
        )
        return False


"""
PackedMV
"""
@dataclass
class PackedMV:
    points: np.ndarray   # (total_vectors, dim) float32, C-contiguous
    offsets: np.ndarray  # (num_items+1,) int32/int64
    
def _choose_offsets_dtype(total_vectors: int) -> np.dtype:
    # offsets 里最大值是 total_vectors
    return np.int32 if total_vectors <= np.iinfo(np.int32).max else np.int64

def pack_list_to_points_offsets(
    seq: List[Union[np.ndarray, torch.Tensor]],
    dim: int,
    points_dtype=np.float32,
) -> PackedMV:
    lengths = np.empty((len(seq),), dtype=np.int64)
    for i, x in enumerate(seq):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError(f"item {i} shape {x.shape} != (L, {dim})")
        lengths[i] = x.shape[0]

    total = int(lengths.sum())
    offsets_dtype = _choose_offsets_dtype(total)
    offsets = np.empty((len(seq) + 1,), dtype=offsets_dtype)
    offsets[0] = 0
    np.cumsum(lengths.astype(offsets_dtype, copy=False), out=offsets[1:])

    points = np.empty((total, dim), dtype=points_dtype)
    cur = 0
    for x in seq:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=points_dtype, order="C")
        L = x.shape[0]
        points[cur:cur + L] = x
        cur += L

    return PackedMV(points=np.ascontiguousarray(points), offsets=offsets)

def save_packed_mv(packed: PackedMV, points_path: str, offsets_path: str) -> None:
    os.makedirs(os.path.dirname(points_path), exist_ok=True)
    # np.save 是最快/最简单的顺序写；不压缩
    np.save(points_path, np.ascontiguousarray(packed.points))
    np.save(offsets_path, np.ascontiguousarray(packed.offsets))

def load_packed_mv(points_path: str, offsets_path: str) -> PackedMV:
    points = np.load(points_path, allow_pickle=False)
    offsets = np.load(offsets_path, allow_pickle=False)
    if points.dtype != np.float32:
        points = points.astype(np.float32, copy=False)
    if not points.flags["C_CONTIGUOUS"]:
        points = np.ascontiguousarray(points)
    if offsets.ndim != 1:
        raise ValueError(f"offsets should be 1D, got {offsets.shape}")
    return PackedMV(points=points, offsets=offsets)

def gather_packed_subset(packed: PackedMV, doc_ids: np.ndarray) -> PackedMV:
    doc_ids = np.asarray(doc_ids, dtype=np.int32)
    offsets = packed.offsets
    points = packed.points

    lens = (offsets[doc_ids + 1] - offsets[doc_ids]).astype(np.int64, copy=False)
    total = int(lens.sum())
    off_dtype = np.int32 if total <= np.iinfo(np.int32).max else np.int64

    new_offsets = np.empty((len(doc_ids) + 1,), dtype=off_dtype)
    new_offsets[0] = 0
    np.cumsum(lens.astype(off_dtype, copy=False), out=new_offsets[1:])

    dim = points.shape[1]
    new_points = np.empty((total, dim), dtype=np.float32)

    cur = 0
    for d in doc_ids:
        s = int(offsets[int(d)])
        e = int(offsets[int(d) + 1])
        L = e - s
        new_points[cur:cur + L] = points[s:e]
        cur += L

    return PackedMV(points=new_points, offsets=new_offsets)



"""
Fast counts calculate
"""
@njit(fastmath=True)
def _gray_code(n):
    """标准的二进制转 Gray Code: (n ^ (n >> 1))"""
    return n ^ (n >> 1)

@njit(parallel=False, fastmath=True)
def _process_sketches_and_count_numba(
    sketches: np.ndarray,      # (total_vectors, num_projections)
    indptr: np.ndarray,        # (num_docs + 1,) 文档偏移量
    num_partitions: int,
    num_docs: int
) -> np.ndarray:
    """
    Numba 核心函数：
    1. 遍历每个文档
    2. 遍历文档中的每个向量
    3. 将向量投影结果 (float) 转为 int 索引 (bit packing)
    4. 直接统计到输出矩阵中
    """
    counts = np.zeros((num_docs, num_partitions), dtype=np.int32)
    
    # 获取投影数量 (hash位数)
    num_projections = sketches.shape[1]
    
    # 遍历所有文档
    for doc_i in range(num_docs):
        start_idx = indptr[doc_i]
        end_idx = indptr[doc_i + 1]
        
        # 遍历文档内的所有向量
        for vec_i in range(start_idx, end_idx):
            # --- 步骤 A: 将 float 投影转为 int 索引 (Bit Packing) ---
            # 相当于原来的 _partition_indices_gray_from_sketches 的核心逻辑
            bits = 0
            for k in range(num_projections):
                if sketches[vec_i, k] > 0:
                    bits |= (1 << k)
            
            # --- 步骤 B: 转换为 Gray Code (如果你的逻辑不需要 Gray，去掉这行即可) ---
            # 注意：这里假设你的 _partition_indices_gray_from_sketches 只是做了标准的 Gray 变换
            idx = _gray_code(bits)
            
            # --- 步骤 C: 统计 (Histogram) ---
            counts[doc_i, idx] += 1
            
    return counts

def _simhash_matrix_from_seed(
    dimension: int, num_projections: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(
        np.float32
    )

def document_partition_counts(
    doc_embeddings: PackedMV,
    config,
    rep_indices = None,
) -> List[np.ndarray]:
    
    rep_loop = list(range(config.num_repetitions)) if rep_indices is None else list(rep_indices)
    all_points = doc_embeddings.points
    indptr = doc_embeddings.offsets
    num_docs = int(indptr.shape[0] - 1)
    
    num_partitions = 2 ** config.num_simhash_projections

    counts_ls = []
    for rep_num in rep_loop:
        seed = config.seed + rep_num
        simhash_matrix = _simhash_matrix_from_seed(
            config.dimension, config.num_simhash_projections, seed
        ).astype(np.float32, copy=False)
        if not simhash_matrix.flags["C_CONTIGUOUS"]:
            simhash_matrix = np.ascontiguousarray(simhash_matrix)

        all_sketches = all_points @ simhash_matrix

        partition_counts = _process_sketches_and_count_numba(
            all_sketches,
            indptr,
            num_partitions,
            num_docs
        )
        counts_ls.append(partition_counts)

    return counts_ls



