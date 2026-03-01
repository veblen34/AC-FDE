import numpy as np


def l2_normalize_np(arr, axis=1, eps=1e-12):
    norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
    return arr / np.clip(norm, eps, None)


def standardize_np(arr, axis=0, eps=1e-12):
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    return (arr - mean) / (std + eps)


def distortion_bound(counts_i):
    counts_i = np.asarray(counts_i)
    inv_d = 1.0 / np.maximum(1, counts_i)
    h = inv_d.mean(axis=0)
    ub_dmin = float(h.min())
    return 2.0 - 2.0 * ub_dmin
