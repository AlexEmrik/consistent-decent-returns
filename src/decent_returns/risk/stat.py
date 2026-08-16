"""Utility functions for decomposing covariance matrices via EM."""

import numpy as np
import pandas as pd
import scipy.sparse.linalg as sp_sparse


def get_F_D_half(
    returns: pd.DataFrame,
    k: int,
    vol_halflife: int = 42,
    cov_halflife: int = 126,
    clip_level: float = 3.0,
    max_iters: int = 10,
    min_samples: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Get the factor loading matrix and half of the diagonal of the covariance matrix.

    Args:
        returns (pd.DataFrame): Returns of shape (n, T).
        k (int): Number of factors.
        vol_halflife (int, optional): Volatility half-life. Defaults to 42.
        cov_halflife (int, optional): Covariance half-life. Defaults to 126.
        clip_level (float, optional): Clip level. Defaults to 3.0.
        max_iters (int, optional): Maximum number of EM iterations. Defaults to 10.
        min_samples (int, optional): Minimum number of samples. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Factor loading matrix and half of the diagonal of
            the covariance matrix.
    """
    whitened_returns, final_vol = whiten_returns(returns, vol_halflife, clip_level, min_samples)
    corr_matrix, assets = get_corr_matrix(whitened_returns, cov_halflife)
    F, D = factor_decomp(corr_matrix, k, max_iters=max_iters)
    F = pd.DataFrame(final_vol.to_numpy()[:, None] * F, index=assets, columns=range(k))
    D_half = pd.Series(final_vol.to_numpy() * np.sqrt(D), index=assets)
    return F, D_half


def whiten_returns(
    returns: pd.DataFrame,
    vol_halflife: int = 42,
    clip_level: float = 3.0,
    min_samples: int = 21,
    eps: float = 1e-4,
) -> tuple[pd.DataFrame, pd.Series]:
    """Whiten returns using a volatility half-life.

    Args:
        returns (pd.DataFrame): Returns of shape (n, T).
        vol_halflife (int, optional): Volatility half-life. Defaults to 42.
        clip_level (float, optional): Clip level. Defaults to 3.0.
        min_samples (int, optional): Minimum number of samples. Defaults to 42.
        eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-4.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Whitened returns of shape (n, T) and final volatility.
    """
    vols = np.sqrt(np.square(returns).ewm(halflife=vol_halflife, min_periods=min_samples).mean())
    returns = returns.div(vols + eps).dropna()
    return np.clip(returns, -clip_level, clip_level), vols.iloc[-1]


def get_corr_matrix(whitened_returns: pd.DataFrame, cov_halflife: int = 126) -> tuple[np.ndarray, pd.Index]:
    """Get the correlation matrix from whitened returns.

    Args:
        whitened_returns (pd.DataFrame): Whitened returns of shape (n, T).
        cov_halflife (int, optional): Covariance half-life. Defaults to 126.

    Returns:
        tuple[np.ndarray, pd.Index]: Correlation matrix and assets.
    """
    decay_factor = np.log(2) / cov_halflife
    weights = np.exp(-decay_factor * np.arange(len(whitened_returns)))[::-1]
    weights = weights / np.sum(weights)

    assets = whitened_returns.columns
    scaled_returns = whitened_returns.mul(np.sqrt(weights), axis=0).to_numpy()
    second_moment = scaled_returns.T @ scaled_returns
    inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(np.diag(second_moment), 1e-8))
    return inv_sqrt_diag[None, :] * second_moment * inv_sqrt_diag[:, None], assets


def factor_decomp(
    S: np.ndarray, k: int, tol: float = 1e-3, max_iters: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize a covariance matrix S into FF^T + D via EM.

    Args:
        S (np.ndarray): Covariance matrix (C_xx = Σ).
        k (int): Rank of the low-rank component F.
        tol (float, optional): Convergence tolerance. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of EM iterations. Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: Factor loading matrix and diagonal matrix.
    """
    eigvals, eigvecs = sp_sparse.eigsh(S, k=k, which="LM")
    F = eigvecs.real * np.sqrt(np.maximum(eigvals, 0))

    residual = S - F @ F.T
    D = np.maximum(np.diag(residual), 1e-8)

    for _ in range(max_iters):
        F_new, D_new = em_iter(S, F, D)

        if stopping_criterion(F, F_new, D, D_new, tol):
            return F_new, D_new

        F, D = F_new, D_new

    return F, D


def em_iter(S: np.ndarray, F: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform one EM iteration.

    Args:
        S (np.ndarray): Covariance matrix (C_xx = Σ).
        F (np.ndarray): Current factor loading matrix of shape (p, k).
        D (np.ndarray): Current diagonal matrix stored in vector form of shape (p,).

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated factor loading matrix and diagonal matrix.
    """
    # E-step:
    d_inv = 1.0 / D
    Ginv = (F.T * d_inv) @ F + np.eye(F.shape[1])
    G = np.linalg.inv(Ginv)
    L = G @ (F.T * d_inv)
    Cxs = S @ L.T
    Css = L @ S @ L.T + G

    # M-step:
    F_new = Cxs @ np.linalg.inv(Css)
    residual = S - 2 * Cxs @ F_new.T + F_new @ Css @ F_new.T
    D_new = np.maximum(np.diag(residual), 1e-8)

    return F_new, D_new


def stopping_criterion(
    F_old: np.ndarray, F_new: np.ndarray, D_old: np.ndarray, D_new: np.ndarray, tol: float
) -> bool:
    """Check if the EM algorithm has converged.

    Args:
        F_old (np.ndarray): Previous factor loading matrix.
        F_new (np.ndarray): Current factor loading matrix.
        D_old (np.ndarray): Previous diagonal matrix.
        D_new (np.ndarray): Current diagonal matrix.
        tol (float): Convergence tolerance.

    Returns:
        bool: True if converged, False otherwise.
    """
    # Relative change in Frobenius norm
    F_change = np.linalg.norm(F_new - F_old) / (np.linalg.norm(F_old) + 1e-10)
    D_change = np.linalg.norm(D_new - D_old) / (np.linalg.norm(D_old) + 1e-10)

    return F_change < tol and D_change < tol
