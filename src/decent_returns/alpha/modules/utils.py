from typing import Literal

import numpy as np


def standard_scale(x: np.ndarray, axis=1) -> np.ndarray:
    numer = x - np.nanmean(x, axis=axis, keepdims=True)
    denom = np.nanstd(x, axis=axis, keepdims=True) + 1e-8
    return numer / denom


def tail_mask(y: np.ndarray, n_keep: int) -> np.ndarray:
    mask = np.zeros_like(y)
    T = y.shape[0]
    order = np.argsort(y, axis=1)
    rows = np.arange(T)[:, None]
    mask[rows, order[:, :n_keep]] = 1
    mask[rows, order[:, -n_keep:]] = 1
    return mask

def rank_cs(y: np.ndarray, axis: int = 1) -> np.ndarray:
    order = np.argsort(np.argsort(y, axis=axis), axis=axis)
    N = y.shape[axis]
    return 2.0 * order / (N - 1) - 1.0

def signed_square(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * (y**2)




def rank(y: np.ndarray, axis=1) -> np.ndarray:
    return np.argsort(np.argsort(y, axis=axis), axis=axis)

def ic_score(
    a1: np.ndarray, a2: np.ndarray, method: Literal["spearman", "pearson"] = "spearman"
) -> np.ndarray:
    assert a1.shape == a2.shape
    T = a1.shape[0]
    ic = np.full((T,), fill_value=np.nan, dtype=float)
    notna = ~np.isnan(a1).any(axis=1) & ~np.isnan(a2).any(axis=1)
    match method:
        case "spearman":
            a1_ = rank(a1[notna])
            a2_ = rank(a2[notna])
        case "pearson":
            a1_ = a1[notna]
            a2_ = a2[notna]
        case _:
            raise ValueError
    a1_ = a1_ - a1_.mean(axis=1, keepdims=True)
    a2_ = a2_ - a2_.mean(axis=1, keepdims=True)

    cov = (a1_ * a2_).sum(axis=1)
    std = np.sqrt(np.nansum(a1_**2, axis=1) * np.nansum(a2_**2, axis=1)) + 1e-8
    ic[notna] = cov / std
    return ic


def rmse(e: np.ndarray, axis=0) -> float:
    return np.sqrt(np.nanmean(e**2, axis=axis))


def mae(e: np.ndarray, axis=0) -> float:
    return np.nanmean(np.abs(e), axis=axis)


def nrmse_score(prd: np.ndarray, ref: np.ndarray, axis: int = 0) -> np.ndarray:
    mu = np.nanmean(ref, axis=axis, keepdims=True)
    nrmse_asset = rmse(prd - ref, axis=axis) / rmse(ref - mu, axis=axis)
    return nrmse_asset

def r2_score(prd: np.ndarray, ref: np.ndarray, axis: int = 0) -> np.ndarray:
    mu = np.nanmean(ref, axis=axis, keepdims=True)
    r2_asset = 1 - np.nansum((prd - ref) ** 2, axis=axis) / np.nansum((ref - mu) ** 2, axis=axis)
    return r2_asset


# def t_test(
#         samples: pd.Series,
#         mu_h0: float = 0.0,
#     ) -> float:
#     mu = samples.mean()
#     sd = samples.std()
#     N = samples.count()
#     t_val = (mu - mu_h0) / (sd / np.sqrt(N))
#     return t_val
