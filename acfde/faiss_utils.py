import faiss
import numpy as np
import torch

from .tools_utils import TrackStep


def fde_matrix_faiss(
    q_fde: np.ndarray,
    d_fde: np.ndarray,
    efs: int = 4000,
    top_k: int = 1000,
):
    with TrackStep("HNSW index building") as t1:
        _, dim = q_fde.shape
        num_docs = d_fde.shape[0]
        print(f"Query: {len(q_fde)}, Corpus shape: {num_docs}, Dim: {dim}")

        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(np.ascontiguousarray(d_fde.astype(np.float32)))
        index.hnsw.efSearch = efs

    print(f"HNSW efSearch = {index.hnsw.efSearch}")
    with TrackStep("HNSW searching") as t2:
        _, indices = index.search(np.ascontiguousarray(q_fde.astype(np.float32)), top_k)

    return indices.tolist(), t1.dt, t2.dt


def fde_matrix_cal_faiss(
    q_fde: np.ndarray,
    d_fde: np.ndarray,
    device: str = "cuda",
    top_k: int = 1000,
    use_gpu: bool = True,
):
    with TrackStep("FlatIP index building") as t1:
        _, dim = q_fde.shape
        num_docs = d_fde.shape[0]
        print(f"Query: {len(q_fde)}, Corpus shape: {num_docs}, Dim: {dim}")

        index = faiss.IndexFlatIP(dim)
        want_gpu = bool(use_gpu and device == "cuda" and torch.cuda.is_available())
        if want_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    print("[WARN] FAISS GPU unavailable; fallback to CPU.")
            except Exception as exc:
                print(f"[WARN] FAISS GPU init failed ({exc}); fallback to CPU.")

        index.add(np.ascontiguousarray(d_fde.astype(np.float32)))

    with TrackStep("FlatIP searching") as t2:
        _, indices = index.search(np.ascontiguousarray(q_fde.astype(np.float32)), top_k)

    return indices.tolist(), t1.dt, t2.dt
