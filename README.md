# AC-FDE (Anonymous Submission)

This repository contains a minimal implementation of the ACFDE pipeline for review.

## Appendix
You can find the theoretical analysis in the appendix of the paper. (Appendix.pdf)

## Environment Setup

We recommend using Conda to manage the environment.

1.  **Create and activate a new environment:**

    ```bash
    conda create -n acfde python=3.11
    conda activate acfde
    ```

2.  **Install dependencies:**

    ```bash
    # Install PyLate (required for ColBERT related tasks)
    pip install pylate
    
    # Install other requirements
    pip install -r requirements.txt
    ```

3.  **Compile the C++ acceleration backend:**

    The `fde_cpp` module is required for the acceleration backend.

    ```bash
    python3 setup_fde_cpp.py build_ext --inplace
    ```

## Data Layout

Dataset root (default `./datasets`):

- `datasets/<dataset>/...` (BEIR-style split files)

Embedding root (default `./output`):

- `output/<dataset>/<sv_encoder>/corpus_sv.pt`
- `output/<dataset>/<sv_encoder>/query_sv.pt` (or `query_sv_train.pt`)
- `output/<dataset>/<mv_encoder>/corpus_mv.pt` (Optional, source format)
- `output/<dataset>/<mv_encoder>/corpus_points.npy` (Packed format)
- `output/<dataset>/<mv_encoder>/corpus_offsets.npy` (Packed format)

## Embedding Format Details

The pipeline supports two formats for multi-vector embeddings. The code will automatically convert the source `.pt` format to the packed `.npy` format if the latter does not exist.

### 1. Source Format: PyTorch List (`corpus_mv.pt`)

-   **File**: `corpus_mv.pt` / `query_mv.pt`
-   **Structure**: A **list of document embeddings**.
-   **Content**: `List[torch.Tensor]` or `List[np.ndarray]`.
    -   Each element is a matrix of shape `(L_i, dim)`, representing the multi-vector embedding for the $i$-th document.

### 2. Packed Format: NumPy Arrays (`points.npy` + `offsets.npy`)

This is the efficient internal format used by AC-FDE.

#### `points.npy`
-   **Shape**: `(total_vectors, dim)`
-   **Dtype**: `float32`
-   **Content**: A flattened 2D array containing all vectors from all documents (or queries) concatenated together.

#### `offsets.npy`
-   **Shape**: `(num_documents + 1,)`
-   **Dtype**: `int64` (or `int32`)
-   **Content**: The starting index of vectors for each document in `points.npy`.
    -   `offsets[i]` is the start index for document `i`.
    -   `offsets[i+1]` is the end index (exclusive) for document `i`.
    -   The number of vectors for document `i` is `offsets[i+1] - offsets[i]`.
    -   `offsets[0]` must be `0`.
    -   `offsets[-1]` must be `total_vectors`.

## Run

```bash
python3 scripts/acfde.py \
  --dataset fiqa \
  --mv-encoder colbert \
  --sv-encoder qwen06b \
  --hybrid \
  --hnsw \
  --dataset-root ./datasets \
  --output-root ./output \
  # --train-fusion \
```

