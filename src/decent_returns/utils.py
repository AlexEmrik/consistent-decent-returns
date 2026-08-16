"""Utility functions."""

import pickle
from pathlib import Path

import pandas as pd


def adjust_returns(
    asset_returns: pd.DataFrame, factor_exposures: pd.DataFrame, factor_returns: pd.DataFrame
) -> pd.DataFrame:
    """Adjust asset returns for factor exposures and factor returns."""
    return asset_returns - factor_returns.dot(factor_exposures.T)


def save_to_pkl(data: dict | tuple, path: Path) -> None:
    """Save data to pickle file.

    Args:
        data (dict | tuple): Data to save.
        path (Path): Path to save the data.
    """
    with path.open("wb") as f:
        pickle.dump(data, f)


def load_from_pkl(path: Path) -> dict | tuple:
    """Load data from pickle file.

    Args:
        path (Path): Path to load the data from.

    Returns:
        dict | tuple: Loaded data.
    """
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301
