"""Model C -- MLP trained directly on the scoring metric.

A and B/D all fit E[y|x] (L2) and only resemble Pearson through that proxy.
This one optimises -Pearson(pred, target) directly, one full moon
cross-section per step (PLAN.md 4, Model C) -- it is free to ignore scale
and fit ranking alone, which should decorrelate it from A/B/D more than
another tree variant would (JOURNAL.md 2026-07-29: corr(B, D) = 0.835).

Features come in as int8 codes 0..6 (PLAN.md 3.1); scaled to roughly
[-0.5, 0.5] here since a plain MLP, unlike LightGBM, is sensitive to input
scale.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_FEATURES = 1150
HIDDEN = (256, 128, 64)
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3
LR = 1e-3
EPOCHS = 60
PATIENCE = 8
ES_SPAN = 52  # moons carved off the tail of train for early stopping
SEED = 42


class MLP(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden=HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        dims = [n_features, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-Pearson correlation, differentiable. eps guards a degenerate (constant) batch."""
    p = pred - pred.mean()
    t = target - target.mean()
    denom = torch.sqrt((p * p).sum() * (t * t).sum()) + 1e-8
    return -(p * t).sum() / denom


def _to_tensor(cache, moon: int) -> tuple[torch.Tensor, torch.Tensor]:
    sl = cache.rows(moon)
    x = cache.X[sl].astype(np.float32) / 6.0 - 0.5
    y = cache.target[sl].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def subsample_moons(moons: np.ndarray, stride: int) -> np.ndarray:
    return moons[::-1][::stride][::-1]


def _epoch_ic(model: MLP, cache, moons: np.ndarray) -> float:
    model.eval()
    ics = []
    with torch.no_grad():
        for m in moons:
            x, y = _to_tensor(cache, int(m))
            if y.std() == 0:
                continue
            pred = model(x)
            if pred.std() == 0:
                ics.append(0.0)
                continue
            ics.append(-pearson_loss(pred, y).item())
    return float(np.mean(ics)) if ics else 0.0


def fit(cache, train_moons: np.ndarray, stride: int = 4, es_span: int = ES_SPAN,
        epochs: int = EPOCHS, patience: int = PATIENCE, hidden=HIDDEN, dropout: float = DROPOUT,
        lr: float = LR, weight_decay: float = WEIGHT_DECAY, seed: int = SEED) -> MLP:
    torch.manual_seed(seed)

    train_moons = np.sort(train_moons)
    if len(train_moons) > es_span + 50:
        fit_moons, es_moons = train_moons[:-es_span], train_moons[-es_span:]
    else:
        fit_moons, es_moons = train_moons, train_moons[-min(len(train_moons), es_span):]
    fit_moons = subsample_moons(fit_moons, stride)

    model = MLP(hidden=hidden, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ic, best_state, bad_epochs = -1e9, None, 0
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(fit_moons))
        for i in order:
            x, y = _to_tensor(cache, int(fit_moons[i]))
            if y.std() == 0:
                continue
            opt.zero_grad()
            loss = pearson_loss(model(x), y)
            loss.backward()
            opt.step()

        es_ic = _epoch_ic(model, cache, es_moons)
        if es_ic > best_ic:
            best_ic, best_state, bad_epochs = es_ic, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_raw(model: MLP, X: np.ndarray) -> np.ndarray:
    """X: (n, n_features) int8 codes 0..6 -> prediction array."""
    model.eval()
    x = torch.from_numpy(X.astype(np.float32) / 6.0 - 0.5)
    with torch.no_grad():
        return model(x).numpy()


def predict(cache, moons: np.ndarray, model: MLP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (prediction, target, moon) stacked over the given moons, for the CV harness."""
    preds, targets, mids = [], [], []
    for m in moons:
        sl = cache.rows(int(m))
        X = cache.X[sl].astype(np.float32)
        preds.append(predict_raw(model, X))
        targets.append(cache.target[sl])
        mids.append(np.full(X.shape[0], int(m), dtype=np.int32))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(mids)
