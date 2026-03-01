import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .evaluate import evaluate
from .faiss_utils import fde_matrix_cal_faiss, fde_matrix_faiss
from .fde_generator import (
    EncodingType,
    FixedDimensionalEncodingConfig,
    ProjectionType,
    build_dfde_pca_streaming_from_packed,
    fde_encode_query_cpp,
    fde_encode_select_packed,
    precompute_fde_matrices_numpy,
)
from .fusion_learning import (
    apply_dimensionwise_fusion,
    build_concat_features,
    train_dimensionwise_fusion,
)
from .math_utils import distortion_bound, l2_normalize_np, standardize_np
from .pca_utils import FastBlockPCA
from .tools_utils import (
    TrackStep,
    document_partition_counts,
    gather_packed_subset,
    load_packed_mv,
    pack_list_to_points_offsets,
    save_packed_mv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"


def _to_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)


def set_rand_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_results(query_key_ls, corpus_key_ls, indices):
    results = {
        query_key_ls[idx]: {
            corpus_key_ls[gid]: float(len(indices[idx]) - rank)
            for rank, gid in enumerate(indices[idx])
        }
        for idx in range(len(indices))
    }

    for key, inner in results.items():
        inner.pop(key, None)
    return results


def _build_split_suffix(query_split: str) -> str:
    return "" if query_split in {"test", "dev"} else f"_{query_split}"


def _ensure_packed_mv(
    dataset_name: str,
    mv_encoder: str,
    query_split: str,
    output_root: Path,
):
    mv_dir = output_root / dataset_name / mv_encoder

    corpus_points = mv_dir / "corpus_points.npy"
    corpus_offsets = mv_dir / "corpus_offsets.npy"
    query_suffix = _build_split_suffix(query_split)
    query_points = mv_dir / f"query_points{query_suffix}.npy"
    query_offsets = mv_dir / f"query_offsets{query_suffix}.npy"

    corpus_mv_pt = mv_dir / "corpus_mv.pt"
    query_mv_pt = mv_dir / f"query_mv{query_suffix}.pt"

    if not corpus_points.exists() or not corpus_offsets.exists():
        if not corpus_mv_pt.exists():
            raise FileNotFoundError(
                f"Missing packed corpus MV ({corpus_points}) and fallback PT ({corpus_mv_pt})."
            )

        corpus_mv_obj = torch.load(corpus_mv_pt, map_location="cpu", weights_only=False)
        if not isinstance(corpus_mv_obj, (list, tuple)) or len(corpus_mv_obj) == 0:
            raise ValueError(f"Invalid corpus MV format in {corpus_mv_pt}")

        first = _to_numpy_f32(corpus_mv_obj[0])
        if first.ndim != 2:
            raise ValueError(f"Expected 2D item in {corpus_mv_pt}, got {first.shape}")

        packed = pack_list_to_points_offsets(list(corpus_mv_obj), dim=first.shape[1])
        mv_dir.mkdir(parents=True, exist_ok=True)
        save_packed_mv(packed, str(corpus_points), str(corpus_offsets))

    if not query_points.exists() or not query_offsets.exists():
        if not query_mv_pt.exists():
            raise FileNotFoundError(
                f"Missing packed query MV ({query_points}) and fallback PT ({query_mv_pt})."
            )

        query_mv_obj = torch.load(query_mv_pt, map_location="cpu", weights_only=False)
        if not isinstance(query_mv_obj, (list, tuple)) or len(query_mv_obj) == 0:
            raise ValueError(f"Invalid query MV format in {query_mv_pt}")

        first = _to_numpy_f32(query_mv_obj[0])
        if first.ndim != 2:
            raise ValueError(f"Expected 2D item in {query_mv_pt}, got {first.shape}")

        packed = pack_list_to_points_offsets(list(query_mv_obj), dim=first.shape[1])
        mv_dir.mkdir(parents=True, exist_ok=True)
        save_packed_mv(packed, str(query_points), str(query_offsets))

    corpus_mv = load_packed_mv(str(corpus_points), str(corpus_offsets))
    query_mv = load_packed_mv(str(query_points), str(query_offsets))
    return corpus_mv, query_mv


def load_embs(
    dataset_name: str,
    mv_encoder: str,
    sv_encoder: str,
    output_root: Path,
    query_split: str = "test",
):
    output_root = Path(output_root)

    sv_dir = output_root / dataset_name / sv_encoder
    mv_dir = output_root / dataset_name / mv_encoder

    split_suffix = _build_split_suffix(query_split)

    corpus_sv_path = sv_dir / "corpus_sv.pt"
    query_sv_path = sv_dir / f"query_sv{split_suffix}.pt"

    if not corpus_sv_path.exists():
        raise FileNotFoundError(f"Missing corpus SV: {corpus_sv_path}")
    if not query_sv_path.exists():
        raise FileNotFoundError(f"Missing query SV: {query_sv_path}")

    corpus_sv = torch.load(corpus_sv_path, map_location="cpu", weights_only=False)
    query_sv = torch.load(query_sv_path, map_location="cpu", weights_only=False)

    corpus_mv, query_mv = _ensure_packed_mv(
        dataset_name=dataset_name,
        mv_encoder=mv_encoder,
        query_split=query_split,
        output_root=output_root,
    )

    corpus_sv = _to_numpy_f32(corpus_sv)
    query_sv = _to_numpy_f32(query_sv)

    return corpus_mv, query_mv, corpus_sv, query_sv


def acfde_pipeline(
    dataset: str,
    sv_encoder: str,
    mv_encoder: str,
    hnsw: bool,
    hybrid: bool = False,
    rerank: bool = False,
    mv_pred: Optional[int] = None,
    choose_rate: Optional[float] = None,
    sampling_rate: float = 0.1,
    efs: int = 2000,
    nrep: int = 160,
    ksim: int = 5,
    proj_d: int = 16,
    topk: int = 1000,
    train_fusion: bool = False,
    fusion_epochs: int = 50,
    fusion_batch_size: int = 256,
    fusion_lr: float = 0.001,
    fusion_num_neg: int = 64,
    dataset_root: Optional[str] = None,
    output_root: Optional[str] = None,
    seed: int = 42,
):
    from beir.datasets.data_loader import GenericDataLoader

    if rerank:
        raise NotImplementedError(
            "rerank path is not included in this public minimal project. "
            "Please run with rerank=False."
        )

    dataset_root = Path(
        dataset_root
        or os.environ.get("ACFDE_DATASET_DIR")
        or DEFAULT_DATASET_ROOT
    )
    output_root = Path(
        output_root
        or os.environ.get("ACFDE_OUTPUT_DIR")
        or DEFAULT_OUTPUT_ROOT
    )

    data_split = "dev" if dataset in {"msmarco_small", "msmarco"} else "test"
    data_folder = dataset_root / dataset
    if not data_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_folder}. "
            "Set --dataset-root or ACFDE_DATASET_DIR correctly."
        )

    corpus_dict, queries, qrels = GenericDataLoader(data_folder=str(data_folder)).load(data_split)

    train_queries, train_qrels = None, None
    if train_fusion:
        _, train_queries, train_qrels = GenericDataLoader(data_folder=str(data_folder)).load("train")

    set_rand_seed(seed)

    query_key_ls = list(queries.keys())
    corpus_key_ls = list(corpus_dict.keys())

    if train_fusion:
        train_query_key_ls = list(train_queries.keys())

    topk_list = [10]

    with TrackStep("STEP0: IO loading embs"):
        corpus_mv, query_mv, corpus_sv, query_sv = load_embs(
            dataset,
            mv_encoder=mv_encoder,
            sv_encoder=sv_encoder,
            output_root=output_root,
            query_split=data_split,
        )

    if len(corpus_key_ls) != int(corpus_sv.shape[0]):
        raise ValueError(
            f"Corpus size mismatch: dataset has {len(corpus_key_ls)} docs, corpus_sv has {corpus_sv.shape[0]} rows."
        )
    if len(query_key_ls) != int(query_sv.shape[0]):
        raise ValueError(
            f"Query size mismatch: split={data_split} has {len(query_key_ls)} queries, query_sv has {query_sv.shape[0]} rows."
        )

    train_query_mv = None
    train_query_sv = None
    if train_fusion:
        with TrackStep("STEP0.1: IO loading train query embs"):
            _, train_query_mv, _, train_query_sv = load_embs(
                dataset,
                mv_encoder=mv_encoder,
                sv_encoder=sv_encoder,
                output_root=output_root,
                query_split="train",
            )

        if len(train_query_key_ls) != int(train_query_sv.shape[0]):
            raise ValueError(
                f"Train query size mismatch: train split has {len(train_query_key_ls)}, query_sv_train has {train_query_sv.shape[0]}."
            )

    num_docs = corpus_sv.shape[0]
    num_queries = query_sv.shape[0]
    sample_n = max(1, int(num_docs * sampling_rate))

    if choose_rate is None:
        raise ValueError("choose_rate must be provided")
    if mv_pred is None:
        raise ValueError("mv_pred must be provided")

    if sample_n <= (2**ksim) * proj_d:
        raise ValueError(
            f"sample_n={sample_n} is too small for ksim={ksim}, proj_d={proj_d}. "
            "Increase sampling_rate or reduce ksim/proj_d."
        )

    sample_idx = np.random.choice(num_docs, size=sample_n, replace=False)
    sample_packed = gather_packed_subset(corpus_mv, sample_idx)

    print("OFFLINE stage")
    config_sel = FixedDimensionalEncodingConfig(
        dimension=query_mv.points.shape[1],
        num_repetitions=nrep,
        num_simhash_projections=ksim,
        encoding_type=EncodingType.AVERAGE,
        projection_type=ProjectionType.AMS_SKETCH,
        projection_dimension=proj_d,
    )

    with TrackStep("STEP1: R selection"):
        d_counts_sample = document_partition_counts(sample_packed, config_sel)
        values = [distortion_bound(d_counts_sample[i]) for i in range(nrep)]
        global_order = np.argsort(values)
        keep_r = max(1, int(nrep * choose_rate))
        rep_sel = global_order[:keep_r].tolist()

    if not hybrid:
        with TrackStep("STEP2: d_fde generation"):
            d_fde = fde_encode_select_packed(
                config_sel,
                corpus_mv,
                is_query=False,
                showbar=True,
                rep_indices=rep_sel,
            )
        d_fde = standardize_np(d_fde, axis=0)

        with TrackStep("STEP3: d_fde bpca fit"):
            bpca = FastBlockPCA(ndim=d_fde.shape[1], n_blocks=keep_r)
            bpca.fit(d_fde, sample_ratio=1.0, keep_dim=mv_pred)

        with TrackStep("STEP4: d_fde bpca transform"):
            d_fde = bpca.transform(d_fde)
    else:
        with TrackStep("STEP2: d_fde sample generation"):
            d_fde_sample = fde_encode_select_packed(
                config_sel,
                sample_packed,
                is_query=False,
                showbar=True,
                rep_indices=rep_sel,
            )
            d_fde_sample = standardize_np(d_fde_sample, axis=0)

        with TrackStep("STEP3: d_fde bpca fit"):
            bpca = FastBlockPCA(ndim=d_fde_sample.shape[1], n_blocks=keep_r)
            bpca.fit(d_fde_sample, sample_ratio=1.0, keep_dim=mv_pred)

        with TrackStep("STEP4: d_fde streaming transform"):
            d_fde = build_dfde_pca_streaming_from_packed(
                config_sel=config_sel,
                bpca=bpca,
                rep_indices_selected=rep_sel,
                norm=True,
                topK=bpca.keep_dim,
                packed=corpus_mv,
                r_chunk=1,
                showbar=False,
            )

    print("ONLINE stage")
    precomputed_mats = precompute_fde_matrices_numpy(config_sel, rep_indices=rep_sel)

    with TrackStep("STEP5: q_fde generation") as t_q:
        q_fde = fde_encode_query_cpp(
            config_sel,
            query_mv,
            rep_indices=rep_sel,
            precomputed_matrices=precomputed_mats,
        )
    query_preprocess_t = t_q.dt

    with TrackStep("STEP6: q_fde bpca transform") as t_qpca:
        q_fde_for_pca = l2_normalize_np(q_fde, axis=1) if hybrid else q_fde
        q_fde_pca = bpca.transform(q_fde_for_pca)
    query_preprocess_t += t_qpca.dt

    if hybrid:
        if train_fusion:
            print("\n=== Training Fusion Module ===")
            with TrackStep("STEP7.train.1: train q_fde generation"):
                train_q_fde = fde_encode_query_cpp(
                    config_sel,
                    train_query_mv,
                    rep_indices=rep_sel,
                    precomputed_matrices=precomputed_mats,
                )
            with TrackStep("STEP7.train.2: train q_fde bpca transform"):
                train_q_fde_norm = l2_normalize_np(train_q_fde, axis=1)
                train_q_fde_pca = bpca.transform(train_q_fde_norm)

            with TrackStep("STEP7.train.3: hard negative mining"):
                hard_topk = min(max(100, topk * 50), int(corpus_sv.shape[0]))
                train_concat_base = np.hstack([train_q_fde_pca, train_query_sv]).astype(np.float32)
                doc_concat_base = np.hstack([d_fde, corpus_sv]).astype(np.float32)
                hard_indices, _, _ = fde_matrix_cal_faiss(
                    train_concat_base,
                    doc_concat_base,
                    top_k=hard_topk,
                )
                hard_neg_by_qidx = {
                    int(i): np.asarray(hard_indices[i], dtype=np.int64)
                    for i in range(len(train_query_key_ls))
                }

            with TrackStep("STEP7.train.4: fusion training"):
                fusion_model, _ = train_dimensionwise_fusion(
                    train_q_fde_pca,
                    train_query_sv,
                    d_fde,
                    corpus_sv,
                    train_qrels=train_qrels,
                    train_query_keys=train_query_key_ls,
                    corpus_keys=corpus_key_ls,
                    num_epochs=fusion_epochs,
                    batch_size=fusion_batch_size,
                    num_negatives=fusion_num_neg,
                    lr=fusion_lr,
                    normalize_parts=False,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose=True,
                    hard_neg_by_qidx=hard_neg_by_qidx,
                )

            concat_qnp, concat_docnp, (fde_w, sv_w) = apply_dimensionwise_fusion(
                q_fde_pca,
                query_sv,
                d_fde,
                corpus_sv,
                model=fusion_model,
                normalize_parts=False,
                l2_after=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"Dimension-wise scales (mean) - FDE: {fde_w:.4f}, SV: {sv_w:.4f}")
        else:
            concat_qnp = build_concat_features(q_fde_pca, query_sv, normalize_parts=False, l2_after=False)
            concat_docnp = build_concat_features(d_fde, corpus_sv, normalize_parts=False, l2_after=False)

        if hnsw:
            indices, _, index_search_t = fde_matrix_faiss(concat_qnp, concat_docnp, efs=efs, top_k=topk)
        else:
            indices, _, index_search_t = fde_matrix_cal_faiss(concat_qnp, concat_docnp, top_k=topk)
    else:
        if hnsw:
            indices, _, index_search_t = fde_matrix_faiss(q_fde_pca, d_fde, efs=efs, top_k=topk)
        else:
            indices, _, index_search_t = fde_matrix_cal_faiss(q_fde_pca, d_fde, top_k=topk)

    pipeline_t = query_preprocess_t + index_search_t

    results = construct_results(query_key_ls, corpus_key_ls, indices)
    recall_result = evaluate(qrels, results, topk_list, ignore_identical_ids=False)

    print(f"time: {pipeline_t:.4f}")
    print(f"QPS: {num_queries / max(pipeline_t, 1e-12):.4f}")

    return recall_result
