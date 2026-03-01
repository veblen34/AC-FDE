import logging
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List

import numpy as np
from tqdm import tqdm
from numba import njit, prange


# =========================================================
# 0) Core definitions
# =========================================================

class EncodingType(Enum):
    DEFAULT_SUM = 0
    AVERAGE = 1


class ProjectionType(Enum):
    DEFAULT_IDENTITY = 0
    AMS_SKETCH = 1


@dataclass
class FixedDimensionalEncodingConfig:
    dimension: int = 128
    num_repetitions: int = 10
    num_simhash_projections: int = 6
    seed: int = 42
    encoding_type: EncodingType = EncodingType.DEFAULT_SUM
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None


# =========================================================
# 1) Utility: Gray-code / simhash / AMS / count-sketch
# =========================================================

def _append_to_gray_code(gray_code: int, bit: bool) -> int:
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))


def _gray_code_to_binary(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


def _simhash_matrix_from_seed(dimension: int, num_projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(np.float32)


def _ams_projection_matrix_from_seed(dimension: int, projection_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    indices = rng.integers(0, projection_dim, size=dimension)
    signs = rng.choice([-1.0, 1.0], size=dimension)
    out[np.arange(dimension), indices] = signs
    return out


def _apply_count_sketch_to_vector(input_vector: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(final_dimension, dtype=np.float32)
    indices = rng.integers(0, final_dimension, size=input_vector.shape[0])
    signs = rng.choice([-1.0, 1.0], size=input_vector.shape[0])
    np.add.at(out, indices, signs * input_vector)
    return out


def _simhash_partition_index_gray(sketch_vector: np.ndarray) -> int:
    partition_index = 0
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index


def _distance_to_simhash_partition(sketch_vector: np.ndarray, partition_index: int) -> int:
    num_projections = sketch_vector.size
    binary_representation = _gray_code_to_binary(partition_index)
    sketch_bits = (sketch_vector > 0).astype(int)
    binary_array = (binary_representation >> np.arange(num_projections - 1, -1, -1)) & 1
    return int(np.sum(sketch_bits != binary_array))


# =========================================================
# 2) Original (non-select) Muvera Generation
# =========================================================

def _generate_fde_internal(
    point_cloud: np.ndarray,
    config: FixedDimensionalEncodingConfig,
) -> tuple:
    """
    原版：返回 (out_fde, partition_counts_ls)
    注意：如果启用 final_projection_dimension，会只返回投影后的 vector（保持你原逻辑风格）。
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(f"Input shape {point_cloud.shape} inconsistent with config dim {config.dimension}.")
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}")

    num_points, original_dim = point_cloud.shape
    num_partitions = 2 ** config.num_simhash_projections

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = original_dim if use_identity_proj else config.projection_dimension
    if not use_identity_proj and (not projection_dim or projection_dim <= 0):
        raise ValueError("projection_dimension is required for non-identity projections.")

    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)

    partition_counts_ls = []

    pc = point_cloud.astype(np.float32, copy=False)

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        sketches = pc @ _simhash_matrix_from_seed(original_dim, config.num_simhash_projections, current_seed)

        if use_identity_proj:
            projected_matrix = pc
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(original_dim, projection_dim, current_seed)
            projected_matrix = pc @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        rep_fde_sum = np.zeros(num_partitions * projection_dim, dtype=np.float32)
        partition_counts = np.zeros(num_partitions, dtype=np.int32)

        partition_indices = np.array([_simhash_partition_index_gray(sketches[i]) for i in range(num_points)])

        for i in range(num_points):
            start_idx = int(partition_indices[i]) * projection_dim
            rep_fde_sum[start_idx:start_idx + projection_dim] += projected_matrix[i]
            partition_counts[int(partition_indices[i])] += 1

        partition_counts_ls.append(partition_counts)

        if config.encoding_type == EncodingType.AVERAGE:
            for i in range(num_partitions):
                start_idx = i * projection_dim
                if partition_counts[i] > 0:
                    rep_fde_sum[start_idx:start_idx + projection_dim] /= partition_counts[i]
                elif config.fill_empty_partitions and num_points > 0:
                    distances = [_distance_to_simhash_partition(sketches[j], i) for j in range(num_points)]
                    nearest_point_idx = int(np.argmin(distances))
                    rep_fde_sum[start_idx:start_idx + projection_dim] = projected_matrix[nearest_point_idx]

        rep_start_index = rep_num * num_partitions * projection_dim
        out_fde[rep_start_index: rep_start_index + rep_fde_sum.size] = rep_fde_sum

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        return _apply_count_sketch_to_vector(out_fde, config.final_projection_dimension, config.seed)

    return out_fde, partition_counts_ls


def generate_query_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    if config.fill_empty_partitions:
        raise ValueError("Query FDE generation does not support 'fill_empty_partitions'.")
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal(point_cloud, query_config)


def generate_document_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    doc_config = replace(config, encoding_type=EncodingType.AVERAGE, fill_empty_partitions=True)
    return _generate_fde_internal(point_cloud, doc_config)


def generate_fde(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig):
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde(point_cloud, config)
    if config.encoding_type == EncodingType.AVERAGE:
        return generate_document_fde(point_cloud, config)
    raise ValueError(f"Unsupported encoding type: {config.encoding_type}")


def generate_document_fde_batch(doc_embeddings_list: List[np.ndarray], config: FixedDimensionalEncodingConfig):
    """
    原版 batch（全 reps）：返回 (out_fdes, partition_counts_ls)
    """
    partition_counts_ls = []
    batch_start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)

    if num_docs == 0:
        logging.warning("[FDE Batch] Empty document list provided")
        return np.array([])

    valid_docs = []
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2:
            logging.warning(f"[FDE Batch] Document {i} invalid ndim={doc.ndim}, skipping")
            continue
        if doc.shape[1] != config.dimension:
            raise ValueError(f"Document {i} dim mismatch: expected {config.dimension}, got {doc.shape[1]}")
        if doc.shape[0] == 0:
            logging.warning(f"[FDE Batch] Document {i} has no vectors, skipping")
            continue
        valid_docs.append(doc.astype(np.float32, copy=False))

    if len(valid_docs) == 0:
        logging.warning("[FDE Batch] No valid documents after filtering")
        return np.array([])

    doc_embeddings_list = valid_docs
    num_docs = len(doc_embeddings_list)

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        projection_dim = config.dimension
    else:
        if not config.projection_dimension or config.projection_dimension <= 0:
            raise ValueError("projection_dimension must be specified for non-identity projections")
        projection_dim = config.projection_dimension

    num_partitions = 2 ** config.num_simhash_projections

    doc_lengths = np.array([len(doc) for doc in doc_embeddings_list], dtype=np.int32)
    total_vectors = int(np.sum(doc_lengths))
    doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
    doc_indices = np.repeat(np.arange(num_docs), doc_lengths)

    all_points = np.vstack(doc_embeddings_list).astype(np.float32, copy=False)

    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)

    for rep_num in tqdm(range(config.num_repetitions)):
        seed = config.seed + rep_num

        simhash_matrix = _simhash_matrix_from_seed(config.dimension, config.num_simhash_projections, seed)
        all_sketches = all_points @ simhash_matrix

        if use_identity_proj:
            projected_points = all_points
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(config.dimension, projection_dim, seed)
            projected_points = all_points @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        bits = (all_sketches > 0).astype(np.uint32)
        partition_indices = np.zeros(total_vectors, dtype=np.uint32)
        for bit_idx in range(config.num_simhash_projections):
            partition_indices = (partition_indices << 1) + (bits[:, bit_idx] ^ (partition_indices & 1))

        rep_fde_sum = np.zeros((num_docs * num_partitions * projection_dim,), dtype=np.float32)
        partition_counts = np.zeros((num_docs, num_partitions), dtype=np.int32)

        np.add.at(partition_counts, (doc_indices, partition_indices), 1)

        doc_part_indices = doc_indices * num_partitions + partition_indices
        base_indices = doc_part_indices * projection_dim
        for d in range(projection_dim):
            np.add.at(rep_fde_sum, base_indices + d, projected_points[:, d])

        rep_fde_sum = rep_fde_sum.reshape(num_docs, num_partitions, projection_dim)

        partition_counts_ls.append(partition_counts)

        counts_3d = partition_counts[:, :, np.newaxis]
        np.divide(rep_fde_sum, counts_3d, out=rep_fde_sum, where=counts_3d > 0)

        if config.fill_empty_partitions:
            empty_docs, empty_parts = np.where(partition_counts == 0)
            if empty_docs.size > 0 and config.num_simhash_projections > 0:
                k = config.num_simhash_projections
                for doc_idx, part_idx in zip(empty_docs, empty_parts):
                    doc_start = doc_boundaries[doc_idx]
                    doc_end = doc_boundaries[doc_idx + 1]
                    doc_sketches = all_sketches[doc_start:doc_end]
                    binary_rep = _gray_code_to_binary(int(part_idx))
                    target_bits = (binary_rep >> np.arange(k - 1, -1, -1)) & 1
                    distances = np.sum((doc_sketches > 0).astype(int) != target_bits, axis=1)
                    nearest_local_idx = int(np.argmin(distances))
                    rep_fde_sum[doc_idx, part_idx, :] = projected_points[doc_start + nearest_local_idx]

        rep_out_start = rep_num * num_partitions * projection_dim
        out_fdes[:, rep_out_start: rep_out_start + num_partitions * projection_dim] = rep_fde_sum.reshape(num_docs, -1)

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        chunk_size = min(100, num_docs)
        chunks = []
        for i in range(0, num_docs, chunk_size):
            j = min(i + chunk_size, num_docs)
            chunks.append(np.array([
                _apply_count_sketch_to_vector(out_fdes[t], config.final_projection_dimension, config.seed)
                for t in range(i, j)
            ], dtype=np.float32))
        out_fdes = np.vstack(chunks)

    total_time = time.perf_counter() - batch_start_time
    logging.info(f"[FDE Batch] Batch generation completed in {total_time:.3f}s | shape={out_fdes.shape}")
    return out_fdes, partition_counts_ls


def fde_encode(config, embeddings, is_query=False, showbar=False, **kwargs):
    counts_ls = []
    fde_out = []

    if not is_query:
        fde_out, counts = generate_document_fde_batch(embeddings, config)
        return fde_out, np.array(counts)

    it = tqdm(embeddings) if showbar else embeddings
    for mat in it:
        if type(mat) is not np.ndarray:
            mat = mat.numpy()
        fde, counts = generate_query_fde(mat, config)
        counts_ls.append(counts)
        fde_out.append(fde)

    return np.stack(fde_out, axis=0), counts_ls


# =========================================================
# 3) Select-repetition generation (list-based)
# =========================================================

def generate_query_fde_select(point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, rep_indices):
    if config.fill_empty_partitions:
        raise ValueError("Query FDE generation does not support 'fill_empty_partitions'.")
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal_select(point_cloud, query_config, rep_indices=rep_indices)


def _generate_fde_internal_select(
    point_cloud: np.ndarray,
    config: FixedDimensionalEncodingConfig,
    rep_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    单条 point cloud 的 select-rep FDE（返回 vector，不返回 counts）
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(f"Input shape {point_cloud.shape} inconsistent with config dim {config.dimension}.")
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}")

    num_points, original_dim = point_cloud.shape
    num_partitions = 2 ** config.num_simhash_projections

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = original_dim if use_identity_proj else config.projection_dimension
    if not use_identity_proj and (not projection_dim or projection_dim <= 0):
        raise ValueError("projection_dimension is required for non-identity projections.")

    if rep_indices is None:
        rep_loop = list(range(config.num_repetitions))
    else:
        rep_loop = list(rep_indices)
        if len(rep_loop) == 0:
            raise ValueError("rep_indices is empty.")
        for r in rep_loop:
            if r < 0 or r >= config.num_repetitions:
                raise ValueError(f"rep_indices out-of-range rep {r}")

    final_fde_dim = len(rep_loop) * num_partitions * projection_dim
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)

    pc = point_cloud.astype(np.float32, copy=False)

    for out_rep_idx, rep_num in enumerate(rep_loop):
        seed = config.seed + rep_num

        sketches = pc @ _simhash_matrix_from_seed(original_dim, config.num_simhash_projections, seed)

        if use_identity_proj:
            projected_matrix = pc
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(original_dim, projection_dim, seed)
            projected_matrix = pc @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        if config.num_simhash_projections == 0:
            partition_indices = np.zeros(num_points, dtype=np.uint32)
        else:
            bits = (sketches > 0).astype(np.uint32)
            partition_indices = np.zeros(num_points, dtype=np.uint32)
            for bit_idx in range(config.num_simhash_projections):
                partition_indices = (partition_indices << 1) + (bits[:, bit_idx] ^ (partition_indices & 1))

        rep_fde_sum = np.zeros(num_partitions * projection_dim, dtype=np.float32)
        partition_counts = np.zeros(num_partitions, dtype=np.int32)

        np.add.at(partition_counts, partition_indices, 1)

        base_indices = partition_indices.astype(np.int64) * int(projection_dim)
        for d in range(projection_dim):
            np.add.at(rep_fde_sum, base_indices + d, projected_matrix[:, d])

        if config.encoding_type == EncodingType.AVERAGE:
            counts_2d = partition_counts[:, None].astype(np.float32)
            rep_fde_sum_2d = rep_fde_sum.reshape(num_partitions, projection_dim)
            np.divide(rep_fde_sum_2d, counts_2d, out=rep_fde_sum_2d, where=counts_2d > 0)

            if config.fill_empty_partitions and num_points > 0:
                empty_parts = np.where(partition_counts == 0)[0]
                if empty_parts.size > 0:
                    sketch_bits = (sketches > 0).astype(int)
                    k = config.num_simhash_projections
                    bit_positions = np.arange(k - 1, -1, -1, dtype=np.int64)
                    for part_idx in empty_parts:
                        binary_rep = _gray_code_to_binary(int(part_idx))
                        target_bits = (binary_rep >> bit_positions) & 1
                        distances = np.sum(sketch_bits != target_bits, axis=1)
                        nearest_point_idx = int(np.argmin(distances))
                        rep_fde_sum_2d[int(part_idx), :] = projected_matrix[nearest_point_idx]

            rep_fde_sum = rep_fde_sum_2d.reshape(-1)

        rep_start = out_rep_idx * num_partitions * projection_dim
        out_fde[rep_start: rep_start + rep_fde_sum.size] = rep_fde_sum

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        return _apply_count_sketch_to_vector(out_fde, config.final_projection_dimension, config.seed)

    return out_fde


def generate_document_fde_batch_select(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    rep_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    list-based 文档 batch select（返回 out_fdes，不返回 counts）
    """
    num_docs = len(doc_embeddings_list)
    if num_docs == 0:
        logging.warning("[FDE Batch Select] Empty document list provided")
        return np.array([])

    valid_docs = []
    for i, doc in enumerate(doc_embeddings_list):
        if not isinstance(doc, np.ndarray):
            doc = doc.numpy()
        if doc.ndim != 2:
            logging.warning(f"[FDE Batch Select] Document {i} invalid ndim={doc.ndim}, skipping")
            continue
        if doc.shape[1] != config.dimension:
            raise ValueError(f"Document {i} dim mismatch: expected {config.dimension}, got {doc.shape[1]}")
        if doc.shape[0] == 0:
            logging.warning(f"[FDE Batch Select] Document {i} empty, skipping")
            continue
        valid_docs.append(doc.astype(np.float32, copy=False))

    if len(valid_docs) == 0:
        logging.warning("[FDE Batch Select] No valid documents after filtering")
        return np.array([])

    doc_embeddings_list = valid_docs
    num_docs = len(doc_embeddings_list)

    if rep_indices is None:
        rep_loop = list(range(config.num_repetitions))
    else:
        rep_loop = list(rep_indices)
        if len(rep_loop) == 0:
            raise ValueError("rep_indices is empty.")
        for r in rep_loop:
            if r < 0 or r >= config.num_repetitions:
                raise ValueError(f"rep_indices out-of-range rep {r}")

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        projection_dim = config.dimension
    else:
        if not config.projection_dimension or config.projection_dimension <= 0:
            raise ValueError("projection_dimension must be specified for non-identity projections")
        projection_dim = config.projection_dimension

    num_partitions = 2 ** config.num_simhash_projections

    doc_lengths = np.array([len(doc) for doc in doc_embeddings_list], dtype=np.int32)
    total_vectors = int(np.sum(doc_lengths))
    doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
    doc_indices = np.repeat(np.arange(num_docs), doc_lengths)

    all_points = np.vstack(doc_embeddings_list).astype(np.float32, copy=False)

    final_fde_dim = len(rep_loop) * num_partitions * projection_dim
    out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)

    for out_rep_idx, rep_num in enumerate(rep_loop):
        seed = config.seed + rep_num

        simhash_matrix = _simhash_matrix_from_seed(config.dimension, config.num_simhash_projections, seed)
        all_sketches = all_points @ simhash_matrix

        if use_identity_proj:
            projected_points = all_points
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(config.dimension, projection_dim, seed)
            projected_points = all_points @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        if config.num_simhash_projections == 0:
            partition_indices = np.zeros(total_vectors, dtype=np.uint32)
        else:
            bits = (all_sketches > 0).astype(np.uint32)
            partition_indices = np.zeros(total_vectors, dtype=np.uint32)
            for bit_idx in range(config.num_simhash_projections):
                partition_indices = (partition_indices << 1) + (bits[:, bit_idx] ^ (partition_indices & 1))

        rep_fde_sum = np.zeros((num_docs * num_partitions * projection_dim,), dtype=np.float32)
        partition_counts = np.zeros((num_docs, num_partitions), dtype=np.int32)

        np.add.at(partition_counts, (doc_indices, partition_indices), 1)

        doc_part_indices = doc_indices.astype(np.int64) * int(num_partitions) + partition_indices.astype(np.int64)
        base_indices = doc_part_indices * int(projection_dim)
        for d in range(projection_dim):
            np.add.at(rep_fde_sum, base_indices + d, projected_points[:, d])

        rep_fde_sum = rep_fde_sum.reshape(num_docs, num_partitions, projection_dim)

        counts_3d = partition_counts[:, :, np.newaxis].astype(np.float32)
        np.divide(rep_fde_sum, counts_3d, out=rep_fde_sum, where=counts_3d > 0)

        if config.fill_empty_partitions:
            empty_docs, empty_parts = np.where(partition_counts == 0)
            if empty_docs.size > 0 and config.num_simhash_projections > 0:
                k = config.num_simhash_projections
                bit_positions = np.arange(k - 1, -1, -1, dtype=np.int64)

                for doc_idx, part_idx in zip(empty_docs, empty_parts):
                    doc_start = int(doc_boundaries[doc_idx])
                    doc_end = int(doc_boundaries[doc_idx + 1])
                    if doc_start == doc_end:
                        continue

                    doc_sketches = all_sketches[doc_start:doc_end]
                    binary_rep = _gray_code_to_binary(int(part_idx))
                    target_bits = (binary_rep >> bit_positions) & 1
                    distances = np.sum((doc_sketches > 0).astype(int) != target_bits, axis=1)
                    nearest_local_idx = int(np.argmin(distances))
                    rep_fde_sum[doc_idx, part_idx, :] = projected_points[doc_start + nearest_local_idx]

        rep_out_start = out_rep_idx * num_partitions * projection_dim
        out_fdes[:, rep_out_start: rep_out_start + num_partitions * projection_dim] = rep_fde_sum.reshape(num_docs, -1)

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        chunk_size = min(100, num_docs)
        chunks = []
        for i in range(0, num_docs, chunk_size):
            j = min(i + chunk_size, num_docs)
            chunks.append(np.array([
                _apply_count_sketch_to_vector(out_fdes[t], config.final_projection_dimension, config.seed)
                for t in range(i, j)
            ], dtype=np.float32))
        out_fdes = np.vstack(chunks)

    return out_fdes


def fde_encode_select(config, embeddings, is_query=False, showbar=False, rep_indices=None, **kwargs):
    """
    list-based select wrapper（保持你原接口）
    - doc: generate_document_fde_batch_select
    - query: 串行 generate_query_fde_select
    """
    if not is_query:
        return generate_document_fde_batch_select(embeddings, config, rep_indices=rep_indices)

    fde_out = []
    it = tqdm(embeddings) if showbar else embeddings
    for mat in it:
        if type(mat) is not np.ndarray:
            mat = mat.numpy()
        fde_out.append(generate_query_fde_select(mat, config, rep_indices=rep_indices))
    return np.stack(fde_out, axis=0)


# =========================================================
# 4) PackedMV + Packed select (v2)
# =========================================================

@dataclass
class PackedMV:
    points: np.ndarray   # (total_vectors, dim) float32
    offsets: np.ndarray  # (num_items+1,) int64/int32


@njit(parallel=True, fastmath=True)
def _accum_sum_count_offsets_2d_packed_v2(
    projected_points: np.ndarray,   # (N, proj_dim) float32
    part_idx: np.ndarray,           # (N,) uint32
    offsets: np.ndarray,            # (D+1,) int64
    num_partitions: int,
    out_sum2d: np.ndarray,          # (D*num_partitions, proj_dim) float32 (已清零)
    out_cnt: np.ndarray             # (D, num_partitions) int32 (已清零)
):
    D = offsets.shape[0] - 1
    proj_dim = projected_points.shape[1]
    for doc in prange(D):
        s = int(offsets[doc])
        e = int(offsets[doc + 1])
        base = doc * num_partitions
        for i in range(s, e):
            p = int(part_idx[i])
            out_cnt[doc, p] += 1
            row = base + p
            for d in range(proj_dim):
                out_sum2d[row, d] += projected_points[i, d]


def generate_fde_batch_select_packed(
    packed: PackedMV,
    config: FixedDimensionalEncodingConfig,
    rep_indices: Optional[List[int]] = None,
    showbar: bool = False,
) -> np.ndarray:
    points = packed.points
    offsets = packed.offsets.astype(np.int64, copy=False)

    if points.ndim != 2:
        raise ValueError(f"packed.points must be 2D, got {points.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"packed.offsets must be 1D, got {offsets.shape}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(points.shape[0]):
        raise ValueError(f"bad offsets: offsets[0]={offsets[0]}, offsets[-1]={offsets[-1]}, N={points.shape[0]}")

    num_docs = int(offsets.shape[0] - 1)
    total_vectors = int(points.shape[0])

    if rep_indices is None:
        rep_loop = list(range(int(config.num_repetitions)))
    else:
        rep_loop = list(map(int, rep_indices))
        if len(rep_loop) == 0:
            raise ValueError("rep_indices is empty.")
        for r in rep_loop:
            if r < 0 or r >= config.num_repetitions:
                raise ValueError(f"rep_indices out-of-range rep {r}")

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        proj_dim = int(config.dimension)
    else:
        proj_dim = int(config.projection_dimension)
        if proj_dim <= 0:
            raise ValueError("projection_dimension must be > 0 for non-identity projections")

    k = int(config.num_simhash_projections)
    num_partitions = 1 << k

    final_fde_dim = len(rep_loop) * num_partitions * proj_dim
    out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)

    part_idx = np.empty((total_vectors,), dtype=np.uint32)
    cnt = np.empty((num_docs, num_partitions), dtype=np.int32)
    sum2d = np.empty((num_docs * num_partitions, proj_dim), dtype=np.float32)

    it = tqdm(rep_loop) if (showbar and tqdm is not None) else rep_loop
    for out_rep_idx, rep_num in enumerate(it):
        seed = int(config.seed) + int(rep_num)

        simhash_matrix = _simhash_matrix_from_seed(int(config.dimension), k, seed).astype(np.float32, copy=False)
        if not simhash_matrix.flags["C_CONTIGUOUS"]:
            simhash_matrix = np.ascontiguousarray(simhash_matrix)
        sketches = points @ simhash_matrix  # (N,k)

        if use_identity_proj:
            projected = points
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams = _ams_projection_matrix_from_seed(int(config.dimension), proj_dim, seed).astype(np.float32, copy=False)
            if not ams.flags["C_CONTIGUOUS"]:
                ams = np.ascontiguousarray(ams)
            projected = points @ ams
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        if k == 0:
            part_idx.fill(0)
        else:
            part_idx.fill(0)
            for bit_idx in range(k):
                bit = (sketches[:, bit_idx] > 0).astype(np.uint32)
                part_idx[:] = (part_idx << 1) + (bit ^ (part_idx & 1))

        cnt.fill(0)
        sum2d.fill(0.0)
        _accum_sum_count_offsets_2d_packed_v2(
            projected.astype(np.float32, copy=False),
            part_idx,
            offsets,
            num_partitions,
            sum2d,
            cnt
        )

        rep_sum = sum2d.reshape(num_docs, num_partitions, proj_dim)

        if config.encoding_type == EncodingType.AVERAGE:
        # if not isquery:
            cnt3 = cnt[:, :, None].astype(np.float32)
            np.divide(rep_sum, cnt3, out=rep_sum, where=cnt3 > 0)

        if getattr(config, "fill_empty_partitions", False):
            empty_docs, empty_parts = np.where(cnt == 0)
            if empty_docs.size > 0 and k > 0:
                bit_positions = np.arange(k - 1, -1, -1, dtype=np.int64)
                for doc_idx, part in zip(empty_docs, empty_parts):
                    s = int(offsets[int(doc_idx)])
                    e = int(offsets[int(doc_idx) + 1])
                    if s == e:
                        continue
                    doc_sketches = sketches[s:e]
                    binary_rep = _gray_code_to_binary(int(part))
                    target_bits = (binary_rep >> bit_positions) & 1
                    distances = np.sum((doc_sketches > 0).astype(np.int8) != target_bits, axis=1)
                    nearest_local = int(np.argmin(distances))
                    rep_sum[int(doc_idx), int(part), :] = projected[s + nearest_local]

        rep_out_start = out_rep_idx * num_partitions * proj_dim
        out_fdes[:, rep_out_start: rep_out_start + num_partitions * proj_dim] = rep_sum.reshape(num_docs, -1)

    if getattr(config, "final_projection_dimension", None) and config.final_projection_dimension and config.final_projection_dimension > 0:
        out_fdes = np.vstack([
            _apply_count_sketch_to_vector(v, config.final_projection_dimension, config.seed)
            for v in out_fdes
        ]).astype(np.float32, copy=False)

    return out_fdes


def fde_encode_query_serial_from_packed(
    config: FixedDimensionalEncodingConfig,
    query_packed: PackedMV,
    showbar: bool = False,
    rep_indices=None,
) -> np.ndarray:
    """
    Query 串行（按 offsets slice）：
    - 不做 batch matmul
    - 每条 query 调 generate_query_fde_select
    """
    points = query_packed.points
    offsets = query_packed.offsets.astype(np.int64, copy=False)

    if points.ndim != 2:
        raise ValueError(f"query_packed.points must be 2D, got {points.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"query_packed.offsets must be 1D, got {offsets.shape}")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(points.shape[0]):
        raise ValueError(f"bad offsets: offsets[0]={offsets[0]}, offsets[-1]={offsets[-1]}, N={points.shape[0]}")

    nq = int(offsets.shape[0] - 1)
    it = tqdm(range(nq)) if (showbar and tqdm is not None) else range(nq)

    out = []
    for i in it:
        s = int(offsets[i])
        e = int(offsets[i + 1])
        pc = points[s:e]  # view
        out.append(generate_query_fde_select(pc, config, rep_indices=rep_indices))
    return np.stack(out, axis=0)


def fde_encode_select_packed(
    config: FixedDimensionalEncodingConfig,
    packed: PackedMV,
    is_query: bool = False,
    showbar: bool = False,
    rep_indices=None,
    **kwargs,
) -> np.ndarray:
    """
    统一入口（Packed）：
    - doc: Packed batch select
    - query: Packed 串行（你要求的行为）
    """
    if not is_query:
        return generate_fde_batch_select_packed(
            packed, config, rep_indices, showbar
        )
    else:
        query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
        return generate_fde_batch_select_packed(
            packed, query_config, rep_indices, showbar
        )

# =========================================================
# 5) STEP3 streaming from Packed (v2)
# =========================================================

def build_dfde_pca_streaming_from_packed(
    config_sel,
    bpca,
    rep_indices_selected,
    norm : bool,
    topK: int,
    packed: PackedMV,
    r_chunk: int = 16,
    showbar: bool = False,
    eps: float = 1e-12,
):
    rep_indices_selected = list(map(int, rep_indices_selected))
    keep_R = len(rep_indices_selected)

    num_docs = int(packed.offsets.shape[0] - 1)
    out = np.empty((num_docs, int(topK)), dtype=bpca.dtype)
    sumsq = np.zeros((num_docs,), dtype=np.float32)

    block_slices = bpca.block_slices

    for b0 in range(0, keep_R, r_chunk):
        b1 = min(keep_R, b0 + r_chunk)
        reps_chunk = rep_indices_selected[b0:b1]

        Xchunk = generate_fde_batch_select_packed(
            packed=packed,
            config=config_sel,
            rep_indices=reps_chunk,
            showbar=showbar,
        ).astype(bpca.dtype, copy=False)

        sumsq += np.einsum("ij,ij->i", Xchunk, Xchunk, dtype=np.float32)

        base_start = block_slices[b0].start
        for b in range(b0, b1):
            sl_g = block_slices[b]
            sl_rel = slice(sl_g.start - base_start, sl_g.stop - base_start)
            Xb = Xchunk[:, sl_rel]

            locs = bpca.kept_local_dims[b]
            outs = bpca.kept_outpos[b]
            if locs.size == 0:
                continue

            P = bpca.P_blocks[b]
            out[:, outs] = (Xb @ P[:, locs]).astype(bpca.dtype, copy=False)
    if norm:
        norm = np.sqrt(np.maximum(sumsq, 0.0)) + eps
        out /= norm[:, None]
    return np.ascontiguousarray(out)


def precompute_fde_matrices(
    config: FixedDimensionalEncodingConfig,
    rep_indices=None,
    device: str = "cuda",
) -> dict:
    import torch
    
    if rep_indices is None:
        rep_loop = list(range(int(config.num_repetitions)))
    else:
        rep_loop = list(map(int, rep_indices))
    
    keep_R = len(rep_loop)
    original_dim = int(config.dimension)
    k = int(config.num_simhash_projections)
    
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    proj_dim = original_dim if use_identity_proj else int(config.projection_dimension)
    
    simhash_matrices = []
    ams_matrices = []
    
    for rep_num in rep_loop:
        seed = int(config.seed) + int(rep_num)
        rng = np.random.default_rng(seed)
        simhash_mat = rng.normal(loc=0.0, scale=1.0, size=(original_dim, k)).astype(np.float32)
        simhash_matrices.append(torch.from_numpy(simhash_mat).to(device, non_blocking=True))
        
        if not use_identity_proj and config.projection_type == ProjectionType.AMS_SKETCH:
            ams_mat = np.zeros((original_dim, proj_dim), dtype=np.float32)
            indices = rng.integers(0, proj_dim, size=original_dim)
            signs = rng.choice([-1.0, 1.0], size=original_dim)
            ams_mat[np.arange(original_dim), indices] = signs
            ams_matrices.append(torch.from_numpy(ams_mat).to(device, non_blocking=True))
    
    return {
        "simhash_stack": torch.stack(simhash_matrices, dim=0),
        "ams_stack": torch.stack(ams_matrices, dim=0) if ams_matrices else None,
    }


def generate_query_fde_cpp(
    packed: "PackedMV",
    config: FixedDimensionalEncodingConfig,
    rep_indices=None,
    precomputed_matrices: dict = None,
) -> np.ndarray:
    try:
        import fde_cpp
    except ImportError:
        # C++ acceleration is required by default for reproducible runtime.
        # Optional debug fallback can be enabled explicitly.
        if os.environ.get("ACFDE_ALLOW_PY_FALLBACK", "0") == "1":
            print(
                "[WARN] fde_cpp is missing; using Python fallback because "
                "ACFDE_ALLOW_PY_FALLBACK=1 (slower)."
            )
            return fde_encode_query_serial_from_packed(
                config=config,
                query_packed=packed,
                showbar=False,
                rep_indices=rep_indices,
            )

        raise ImportError(
            "fde_cpp is required by default. Build it first: "
            "python3 setup_fde_cpp.py build_ext --inplace. "
            "If you intentionally want slower Python fallback, set "
            "ACFDE_ALLOW_PY_FALLBACK=1."
        )
    
    import torch
    
    points = packed.points
    offsets = packed.offsets
    
    if isinstance(offsets, torch.Tensor):
        offsets_np = offsets.cpu().numpy()
    else:
        offsets_np = offsets
    
    num_queries = int(offsets_np.shape[0] - 1)
    total_vectors = points.shape[0] if isinstance(points, torch.Tensor) else int(points.shape[0])
    original_dim = int(config.dimension)
    
    if rep_indices is None:
        rep_loop = list(range(int(config.num_repetitions)))
    else:
        rep_loop = list(map(int, rep_indices))
    
    keep_R = len(rep_loop)
    k = int(config.num_simhash_projections)
    
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    proj_dim = original_dim if use_identity_proj else int(config.projection_dimension)
    
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    if precomputed_matrices is not None:
        simhash_stack = precomputed_matrices["simhash_stack"]
        ams_stack = precomputed_matrices.get("ams_stack", None)
        
        if isinstance(simhash_stack, torch.Tensor):
            simhash_stack = simhash_stack.cpu().numpy()
        if ams_stack is not None and isinstance(ams_stack, torch.Tensor):
            ams_stack = ams_stack.cpu().numpy()
    else:
        simhash_matrices = []
        ams_matrices = []
        
        for rep_num in rep_loop:
            seed = int(config.seed) + int(rep_num)
            rng = np.random.default_rng(seed)
            simhash_mat = rng.normal(loc=0.0, scale=1.0, size=(original_dim, k)).astype(np.float32)
            simhash_matrices.append(simhash_mat)
            
            if not use_identity_proj and config.projection_type == ProjectionType.AMS_SKETCH:
                ams_mat = np.zeros((original_dim, proj_dim), dtype=np.float32)
                indices = rng.integers(0, proj_dim, size=original_dim)
                signs = rng.choice([-1.0, 1.0], size=original_dim)
                ams_mat[np.arange(original_dim), indices] = signs
                ams_matrices.append(ams_mat)
        
        simhash_stack = np.stack(simhash_matrices, axis=0)
        ams_stack = np.stack(ams_matrices, axis=0) if ams_matrices else None
    
    result = fde_cpp.fde_query_cpp(
        points_np.astype(np.float32),
        offsets_np.astype(np.int64),
        simhash_stack.astype(np.float32),
        ams_stack.astype(np.float32) if ams_stack is not None else None,
        num_queries,
        total_vectors,
        original_dim,
        keep_R,
        k,
        proj_dim
    )
    
    return result


def fde_encode_query_cpp(
    config: FixedDimensionalEncodingConfig,
    packed: "PackedMV",
    rep_indices=None,
    precomputed_matrices: dict = None,
) -> np.ndarray:
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return generate_query_fde_cpp(
        packed, query_config, rep_indices, precomputed_matrices
    )


def precompute_fde_matrices_numpy(
    config: FixedDimensionalEncodingConfig,
    rep_indices=None,
) -> dict:
    if rep_indices is None:
        rep_loop = list(range(int(config.num_repetitions)))
    else:
        rep_loop = list(map(int, rep_indices))
    
    keep_R = len(rep_loop)
    original_dim = int(config.dimension)
    k = int(config.num_simhash_projections)
    
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    proj_dim = original_dim if use_identity_proj else int(config.projection_dimension)
    
    simhash_matrices = []
    ams_matrices = []
    
    for rep_num in rep_loop:
        seed = int(config.seed) + int(rep_num)
        simhash_mat = _simhash_matrix_from_seed(original_dim, k, seed)
        simhash_matrices.append(simhash_mat)
        
        if not use_identity_proj and config.projection_type == ProjectionType.AMS_SKETCH:
            ams_mat = _ams_projection_matrix_from_seed(original_dim, proj_dim, seed)
            ams_matrices.append(ams_mat)
    
    return {
        "simhash_stack": np.stack(simhash_matrices, axis=0),
        "ams_stack": np.stack(ams_matrices, axis=0) if ams_matrices else None,
    }
