# AC-FDE (Anonymous Submission)

This repository contains a minimal implementation of the ACFDE pipeline for review.
Only the ACFDE path is included.

## Quick Start

```bash
cd AC-FDE
pip install -r requirements.txt
python3 setup_fde_cpp.py build_ext --inplace
```

`fde_cpp` is the default and required acceleration backend.

## Data Layout

Dataset root (default `./datasets`):

- `datasets/<dataset>/...` (BEIR-style split files)

Embedding root (default `./output`):

- `output/<dataset>/<sv_encoder>/corpus_sv.pt`
- `output/<dataset>/<sv_encoder>/query_sv.pt` (or `query_sv_train.pt`)
- `output/<dataset>/<mv_encoder>/corpus_points.npy`
- `output/<dataset>/<mv_encoder>/corpus_offsets.npy`
- `output/<dataset>/<mv_encoder>/query_points.npy`
- `output/<dataset>/<mv_encoder>/query_offsets.npy`

If only `corpus_mv.pt` / `query_mv.pt` exist, the code converts them to `points+offsets` automatically.

## Run

```bash
python3 scripts/acfde.py \
  --dataset msmarco \
  --mv-encoder colbert \
  --sv-encoder qwen06b \
  --hybrid \
  --hnsw \
  --train-fusion
```

Path overrides:

- `--dataset-root` or `ACFDE_DATASET_DIR`
- `--output-root` or `ACFDE_OUTPUT_DIR`

## Note

For debugging only, slower Python query encoding can be enabled with:

```bash
export ACFDE_ALLOW_PY_FALLBACK=1
```
