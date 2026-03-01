import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(arr)


def l2_normalize_np(arr, axis=1, eps=1e-12):
    norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
    return arr / np.clip(norm, eps, None)


def build_concat_features(part_a, part_b, normalize_parts=True, l2_after=True):
    a = to_numpy_f32(part_a)
    b = to_numpy_f32(part_b)
    if normalize_parts:
        a = l2_normalize_np(a, axis=1)
        b = l2_normalize_np(b, axis=1)
    fused = np.hstack([a, b]).astype(np.float32, copy=False)
    if l2_after:
        fused = l2_normalize_np(fused, axis=1)
    return np.ascontiguousarray(fused)


class LearnableTwoPartFusion(nn.Module):
    def __init__(self, init_scale_a=1.0, init_scale_b=1.0):
        super().__init__()
        init = torch.tensor([float(init_scale_a), float(init_scale_b)], dtype=torch.float32)
        # inverse softplus to make initial positive scales close to init values
        inv_softplus = torch.log(torch.expm1(init.clamp_min(1e-4)))
        self.raw_scales = nn.Parameter(inv_softplus)

    def positive_scales(self):
        return F.softplus(self.raw_scales) + 1e-6

    def get_scales(self):
        with torch.no_grad():
            s = self.positive_scales().detach().cpu().numpy().astype(np.float32)
        return float(s[0]), float(s[1])


def apply_fusion_scales(part_a, part_b, scales, normalize_parts=True, l2_after=True):
    a = to_numpy_f32(part_a)
    b = to_numpy_f32(part_b)
    if normalize_parts:
        a = l2_normalize_np(a, axis=1)
        b = l2_normalize_np(b, axis=1)
    sa, sb = float(scales[0]), float(scales[1])
    fused = np.hstack([sa * a, sb * b]).astype(np.float32, copy=False)
    if l2_after:
        fused = l2_normalize_np(fused, axis=1)
    return np.ascontiguousarray(fused)


def _build_positive_lists(qrels, query_key_ls, corpus_key_ls):
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(corpus_key_ls)}
    pos_lists = []
    train_qidx = []

    for qidx, qid in enumerate(query_key_ls):
        rel = qrels.get(qid, {})
        pos = [doc_to_idx[d] for d, score in rel.items() if score > 0 and d in doc_to_idx]
        if pos:
            train_qidx.append(qidx)
            pos_lists.append(np.asarray(pos, dtype=np.int64))

    if not train_qidx:
        raise ValueError("No positive train pairs were found in qrels.")

    return np.asarray(train_qidx, dtype=np.int64), pos_lists


def train_two_part_fusion(
    query_part_a,
    query_part_b,
    doc_part_a,
    doc_part_b,
    train_qrels,
    train_query_keys,
    corpus_keys,
    num_epochs=30,
    batch_size=256,
    num_negatives=64,
    lr=1e-3,
    weight_decay=1e-4,
    normalize_parts=True,
    device=None,
    seed=42,
    verbose=True,
):
    q_a = to_numpy_f32(query_part_a)
    q_b = to_numpy_f32(query_part_b)
    d_a = to_numpy_f32(doc_part_a)
    d_b = to_numpy_f32(doc_part_b)

    if normalize_parts:
        q_a = l2_normalize_np(q_a, axis=1)
        q_b = l2_normalize_np(q_b, axis=1)
        d_a = l2_normalize_np(d_a, axis=1)
        d_b = l2_normalize_np(d_b, axis=1)

    if q_a.shape[0] != q_b.shape[0]:
        raise ValueError(f"Query parts have different rows: {q_a.shape} vs {q_b.shape}")
    if d_a.shape[0] != d_b.shape[0]:
        raise ValueError(f"Doc parts have different rows: {d_a.shape} vs {d_b.shape}")
    if len(train_query_keys) != q_a.shape[0]:
        raise ValueError(
            f"train_query_keys size ({len(train_query_keys)}) mismatches query rows ({q_a.shape[0]})"
        )
    if len(corpus_keys) != d_a.shape[0]:
        raise ValueError(
            f"corpus_keys size ({len(corpus_keys)}) mismatches doc rows ({d_a.shape[0]})"
        )

    train_qidx, pos_lists = _build_positive_lists(train_qrels, train_query_keys, corpus_keys)
    pos_sets = [set(p.tolist()) for p in pos_lists]
    n_docs = d_a.shape[0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LearnableTwoPartFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    q_a_t = torch.from_numpy(q_a).to(device=device, dtype=torch.float32)
    q_b_t = torch.from_numpy(q_b).to(device=device, dtype=torch.float32)

    losses = []
    for epoch in range(num_epochs):
        perm = rng.permutation(len(train_qidx))
        epoch_loss = 0.0
        batch_count = 0

        for st in range(0, len(perm), batch_size):
            bsel = perm[st : st + batch_size]
            q_index = train_qidx[bsel]

            pos_idx = np.empty((len(q_index),), dtype=np.int64)
            neg_idx = np.empty((len(q_index), num_negatives), dtype=np.int64)

            for i, global_qidx in enumerate(q_index):
                local_pos = pos_lists[bsel[i]]
                pos_idx[i] = local_pos[rng.integers(0, len(local_pos))]
                pset = pos_sets[bsel[i]]

                filled = 0
                while filled < num_negatives:
                    candidate = int(rng.integers(0, n_docs))
                    if candidate not in pset:
                        neg_idx[i, filled] = candidate
                        filled += 1

            qa = q_a_t[q_index]
            qb = q_b_t[q_index]

            pa = torch.from_numpy(d_a[pos_idx]).to(device=device, dtype=torch.float32)
            pb = torch.from_numpy(d_b[pos_idx]).to(device=device, dtype=torch.float32)
            na = torch.from_numpy(d_a[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32)
            nb = torch.from_numpy(d_b[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32)
            na = na.view(len(q_index), num_negatives, -1)
            nb = nb.view(len(q_index), num_negatives, -1)

            scales = model.positive_scales()
            sa2 = scales[0] * scales[0]
            sb2 = scales[1] * scales[1]

            pos_dot_a = (qa * pa).sum(dim=-1)
            pos_dot_b = (qb * pb).sum(dim=-1)
            neg_dot_a = torch.einsum("bd,bnd->bn", qa, na)
            neg_dot_b = torch.einsum("bd,bnd->bn", qb, nb)

            pos_score = sa2 * pos_dot_a + sb2 * pos_dot_b
            neg_score = sa2 * neg_dot_a + sb2 * neg_dot_b
            pair_loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

            reg = 1e-4 * ((scales - 1.0) ** 2).sum()
            loss = pair_loss + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)
        if verbose:
            sa, sb = model.get_scales()
            print(f"[fusion][epoch {epoch + 1:02d}/{num_epochs}] loss={avg_loss:.6f} scale_a={sa:.4f} scale_b={sb:.4f}")

    return model, {"train_loss": losses}


class QueryAdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _query_stats(self, qa, qb):
        # Stats are simple and cheap, but expressive enough to learn per-query mixing.
        qa_abs = qa.abs().mean(dim=-1, keepdim=True)
        qb_abs = qb.abs().mean(dim=-1, keepdim=True)
        qa_l2 = torch.sqrt((qa * qa).sum(dim=-1, keepdim=True) + 1e-12)
        qb_l2 = torch.sqrt((qb * qb).sum(dim=-1, keepdim=True) + 1e-12)
        return torch.cat([qa_abs, qb_abs, qa_l2, qb_l2], dim=-1)

    def query_weights(self, qa, qb):
        stats = self._query_stats(qa, qb)
        # Keep weights positive but allow moderate query-adaptive variation.
        return 0.5 + 1.0 * torch.sigmoid(self.mlp(stats))

    def mean_weights(self, qa, qb):
        with torch.no_grad():
            w = self.query_weights(qa, qb).mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return float(w[0]), float(w[1])


def train_query_adaptive_fusion(
    query_part_a,
    query_part_b,
    doc_part_a,
    doc_part_b,
    train_qrels,
    train_query_keys,
    corpus_keys,
    num_epochs=30,
    batch_size=256,
    num_negatives=64,
    lr=1e-3,
    weight_decay=1e-4,
    normalize_parts=False,
    device=None,
    seed=42,
    hidden_dim=32,
    verbose=True,
    hard_neg_by_qidx=None,
):
    q_a = to_numpy_f32(query_part_a)
    q_b = to_numpy_f32(query_part_b)
    d_a = to_numpy_f32(doc_part_a)
    d_b = to_numpy_f32(doc_part_b)

    if normalize_parts:
        q_a = l2_normalize_np(q_a, axis=1)
        q_b = l2_normalize_np(q_b, axis=1)
        d_a = l2_normalize_np(d_a, axis=1)
        d_b = l2_normalize_np(d_b, axis=1)

    if q_a.shape[0] != q_b.shape[0]:
        raise ValueError(f"Query parts have different rows: {q_a.shape} vs {q_b.shape}")
    if d_a.shape[0] != d_b.shape[0]:
        raise ValueError(f"Doc parts have different rows: {d_a.shape} vs {d_b.shape}")
    if len(train_query_keys) != q_a.shape[0]:
        raise ValueError(
            f"train_query_keys size ({len(train_query_keys)}) mismatches query rows ({q_a.shape[0]})"
        )
    if len(corpus_keys) != d_a.shape[0]:
        raise ValueError(
            f"corpus_keys size ({len(corpus_keys)}) mismatches doc rows ({d_a.shape[0]})"
        )

    train_qidx, pos_lists = _build_positive_lists(train_qrels, train_query_keys, corpus_keys)
    pos_sets = [set(p.tolist()) for p in pos_lists]
    n_docs = d_a.shape[0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QueryAdaptiveFusion(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    q_a_t = torch.from_numpy(q_a).to(device=device, dtype=torch.float32)
    q_b_t = torch.from_numpy(q_b).to(device=device, dtype=torch.float32)

    losses = []
    for epoch in range(num_epochs):
        perm = rng.permutation(len(train_qidx))
        epoch_loss = 0.0
        batch_count = 0

        for st in range(0, len(perm), batch_size):
            bsel = perm[st : st + batch_size]
            q_index = train_qidx[bsel]

            pos_idx = np.empty((len(q_index),), dtype=np.int64)
            neg_idx = np.empty((len(q_index), num_negatives), dtype=np.int64)

            for i, _ in enumerate(q_index):
                local_pos = pos_lists[bsel[i]]
                pos_idx[i] = local_pos[rng.integers(0, len(local_pos))]
                pset = pos_sets[bsel[i]]
                hard_pool = None
                global_qidx = int(q_index[i])
                if hard_neg_by_qidx is not None:
                    hard_pool = hard_neg_by_qidx.get(global_qidx, None)

                filled = 0
                max_try = num_negatives * 20
                tries = 0
                while filled < num_negatives and tries < max_try:
                    if hard_pool is not None and len(hard_pool) > 0 and rng.random() < 0.8:
                        candidate = int(hard_pool[rng.integers(0, len(hard_pool))])
                    else:
                        candidate = int(rng.integers(0, n_docs))
                    tries += 1
                    if candidate not in pset:
                        neg_idx[i, filled] = candidate
                        filled += 1

                while filled < num_negatives:
                    candidate = int(rng.integers(0, n_docs))
                    if candidate not in pset:
                        neg_idx[i, filled] = candidate
                        filled += 1

            qa = q_a_t[q_index]
            qb = q_b_t[q_index]

            pa = torch.from_numpy(d_a[pos_idx]).to(device=device, dtype=torch.float32)
            pb = torch.from_numpy(d_b[pos_idx]).to(device=device, dtype=torch.float32)
            na = torch.from_numpy(d_a[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32)
            nb = torch.from_numpy(d_b[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32)
            na = na.view(len(q_index), num_negatives, -1)
            nb = nb.view(len(q_index), num_negatives, -1)

            w = model.query_weights(qa, qb)
            w0 = w[:, 0]
            w1 = w[:, 1]

            pos_dot_a = (qa * pa).sum(dim=-1)
            pos_dot_b = (qb * pb).sum(dim=-1)
            neg_dot_a = torch.einsum("bd,bnd->bn", qa, na)
            neg_dot_b = torch.einsum("bd,bnd->bn", qb, nb)

            pos_score = w0 * pos_dot_a + w1 * pos_dot_b
            neg_score = w0.unsqueeze(1) * neg_dot_a + w1.unsqueeze(1) * neg_dot_b
            pair_loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

            reg = 2e-4 * ((w - 1.0) ** 2).mean()
            loss = pair_loss + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)
        if verbose:
            m0, m1 = model.mean_weights(q_a_t[train_qidx[: min(len(train_qidx), 2048)]], q_b_t[train_qidx[: min(len(train_qidx), 2048)]])
            print(f"[adaptive_fusion][epoch {epoch + 1:02d}/{num_epochs}] loss={avg_loss:.6f} mean_w_a={m0:.4f} mean_w_b={m1:.4f}")

    return model, {"train_loss": losses}


def apply_query_adaptive_fusion(
    query_part_a,
    query_part_b,
    doc_part_a,
    doc_part_b,
    model,
    normalize_parts=False,
    l2_after=False,
    device=None,
    batch_size=4096,
):
    q_a = to_numpy_f32(query_part_a)
    q_b = to_numpy_f32(query_part_b)
    d_a = to_numpy_f32(doc_part_a)
    d_b = to_numpy_f32(doc_part_b)

    if normalize_parts:
        q_a = l2_normalize_np(q_a, axis=1)
        q_b = l2_normalize_np(q_b, axis=1)
        d_a = l2_normalize_np(d_a, axis=1)
        d_b = l2_normalize_np(d_b, axis=1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    weights = []
    with torch.no_grad():
        for st in range(0, q_a.shape[0], batch_size):
            qa_t = torch.from_numpy(q_a[st : st + batch_size]).to(device=device, dtype=torch.float32)
            qb_t = torch.from_numpy(q_b[st : st + batch_size]).to(device=device, dtype=torch.float32)
            w = model.query_weights(qa_t, qb_t).detach().cpu().numpy().astype(np.float32)
            weights.append(w)
    w_all = np.vstack(weights)

    q_fused = np.hstack([w_all[:, 0:1] * q_a, w_all[:, 1:2] * q_b]).astype(np.float32, copy=False)
    d_fused = np.hstack([d_a, d_b]).astype(np.float32, copy=False)

    if l2_after:
        q_fused = l2_normalize_np(q_fused, axis=1)
        d_fused = l2_normalize_np(d_fused, axis=1)

    mean_w = w_all.mean(axis=0)
    return np.ascontiguousarray(q_fused), np.ascontiguousarray(d_fused), (float(mean_w[0]), float(mean_w[1]))


class DimensionwiseFusion(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.raw_a = nn.Parameter(torch.zeros(dim_a, dtype=torch.float32))
        self.raw_b = nn.Parameter(torch.zeros(dim_b, dtype=torch.float32))

    def scales(self):
        # 0.5 ~ 1.5, initialized at 1.0
        sa = 0.5 + torch.sigmoid(self.raw_a)
        sb = 0.5 + torch.sigmoid(self.raw_b)
        return sa, sb

    def mean_scales(self):
        with torch.no_grad():
            sa, sb = self.scales()
            return float(sa.mean().cpu()), float(sb.mean().cpu())


def train_dimensionwise_fusion(
    query_part_a,
    query_part_b,
    doc_part_a,
    doc_part_b,
    train_qrels,
    train_query_keys,
    corpus_keys,
    num_epochs=10,
    batch_size=256,
    num_negatives=64,
    lr=2e-3,
    weight_decay=1e-5,
    normalize_parts=False,
    device=None,
    seed=42,
    verbose=True,
    hard_neg_by_qidx=None,
):
    q_a = to_numpy_f32(query_part_a)
    q_b = to_numpy_f32(query_part_b)
    d_a = to_numpy_f32(doc_part_a)
    d_b = to_numpy_f32(doc_part_b)

    if normalize_parts:
        q_a = l2_normalize_np(q_a, axis=1)
        q_b = l2_normalize_np(q_b, axis=1)
        d_a = l2_normalize_np(d_a, axis=1)
        d_b = l2_normalize_np(d_b, axis=1)

    train_qidx, pos_lists = _build_positive_lists(train_qrels, train_query_keys, corpus_keys)
    pos_sets = [set(p.tolist()) for p in pos_lists]
    n_docs = d_a.shape[0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DimensionwiseFusion(dim_a=q_a.shape[1], dim_b=q_b.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    q_a_t = torch.from_numpy(q_a).to(device=device, dtype=torch.float32)
    q_b_t = torch.from_numpy(q_b).to(device=device, dtype=torch.float32)

    losses = []
    for epoch in range(num_epochs):
        perm = rng.permutation(len(train_qidx))
        epoch_loss = 0.0
        batch_count = 0

        for st in range(0, len(perm), batch_size):
            bsel = perm[st : st + batch_size]
            q_index = train_qidx[bsel]

            pos_idx = np.empty((len(q_index),), dtype=np.int64)
            neg_idx = np.empty((len(q_index), num_negatives), dtype=np.int64)

            for i, _ in enumerate(q_index):
                local_pos = pos_lists[bsel[i]]
                pos_idx[i] = local_pos[rng.integers(0, len(local_pos))]
                pset = pos_sets[bsel[i]]
                hard_pool = None
                global_qidx = int(q_index[i])
                if hard_neg_by_qidx is not None:
                    hard_pool = hard_neg_by_qidx.get(global_qidx, None)

                filled = 0
                max_try = num_negatives * 20
                tries = 0
                while filled < num_negatives and tries < max_try:
                    if hard_pool is not None and len(hard_pool) > 0 and rng.random() < 0.8:
                        candidate = int(hard_pool[rng.integers(0, len(hard_pool))])
                    else:
                        candidate = int(rng.integers(0, n_docs))
                    tries += 1
                    if candidate not in pset:
                        neg_idx[i, filled] = candidate
                        filled += 1
                while filled < num_negatives:
                    candidate = int(rng.integers(0, n_docs))
                    if candidate not in pset:
                        neg_idx[i, filled] = candidate
                        filled += 1

            qa = q_a_t[q_index]
            qb = q_b_t[q_index]
            pa = torch.from_numpy(d_a[pos_idx]).to(device=device, dtype=torch.float32)
            pb = torch.from_numpy(d_b[pos_idx]).to(device=device, dtype=torch.float32)
            na = torch.from_numpy(d_a[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32).view(len(q_index), num_negatives, -1)
            nb = torch.from_numpy(d_b[neg_idx.reshape(-1)]).to(device=device, dtype=torch.float32).view(len(q_index), num_negatives, -1)

            sa, sb = model.scales()
            qa_s = qa * sa
            qb_s = qb * sb
            pa_s = pa * sa
            pb_s = pb * sb
            na_s = na * sa.unsqueeze(0).unsqueeze(0)
            nb_s = nb * sb.unsqueeze(0).unsqueeze(0)

            pos_score = (qa_s * pa_s).sum(dim=-1) + (qb_s * pb_s).sum(dim=-1)
            neg_score = torch.einsum("bd,bnd->bn", qa_s, na_s) + torch.einsum("bd,bnd->bn", qb_s, nb_s)
            pair_loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()

            reg = 1e-3 * (((sa - 1.0) ** 2).mean() + ((sb - 1.0) ** 2).mean())
            loss = pair_loss + reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)
        if verbose:
            ma, mb = model.mean_scales()
            print(f"[dim_fusion][epoch {epoch + 1:02d}/{num_epochs}] loss={avg_loss:.6f} mean_scale_a={ma:.4f} mean_scale_b={mb:.4f}")

    return model, {"train_loss": losses}


def apply_dimensionwise_fusion(query_part_a, query_part_b, doc_part_a, doc_part_b, model, normalize_parts=False, l2_after=False, device=None):
    q_a = to_numpy_f32(query_part_a)
    q_b = to_numpy_f32(query_part_b)
    d_a = to_numpy_f32(doc_part_a)
    d_b = to_numpy_f32(doc_part_b)

    if normalize_parts:
        q_a = l2_normalize_np(q_a, axis=1)
        q_b = l2_normalize_np(q_b, axis=1)
        d_a = l2_normalize_np(d_a, axis=1)
        d_b = l2_normalize_np(d_b, axis=1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sa_t, sb_t = model.scales()
        sa = sa_t.detach().cpu().numpy().astype(np.float32)
        sb = sb_t.detach().cpu().numpy().astype(np.float32)

    q_fused = np.hstack([q_a * sa[None, :], q_b * sb[None, :]]).astype(np.float32, copy=False)
    d_fused = np.hstack([d_a * sa[None, :], d_b * sb[None, :]]).astype(np.float32, copy=False)

    if l2_after:
        q_fused = l2_normalize_np(q_fused, axis=1)
        d_fused = l2_normalize_np(d_fused, axis=1)

    return np.ascontiguousarray(q_fused), np.ascontiguousarray(d_fused), (float(sa.mean()), float(sb.mean()))
