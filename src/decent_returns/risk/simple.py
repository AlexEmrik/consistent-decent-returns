"""Simple methods for computing covariance matrices."""

import numpy as np
import pandas as pd


def ewma_cov(returns: pd.DataFrame, cov_halflife: int = 126) -> pd.DataFrame:
    """Compute the EWMA covariance matrix.

    Args:
        returns (pd.DataFrame): Returns of shape (n, T).
        cov_halflife (int, optional): Covariance half-life. Defaults to 126.

    Returns:
        pd.DataFrame: EWMA covariance matrix.
    """
    decay_factor = np.log(2) / cov_halflife
    weights = np.exp(-decay_factor * np.arange(len(returns)))[::-1]
    weights = weights / np.sum(weights)

    assets = returns.columns
    scaled_returns = returns.mul(np.sqrt(weights), axis=0).to_numpy()
    second_moment = scaled_returns.T @ scaled_returns
    return pd.DataFrame(second_moment, index=assets, columns=assets)
