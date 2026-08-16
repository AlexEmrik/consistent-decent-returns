"""Load and preprocess the data."""

import io
from pathlib import Path

import pandas as pd


def load_factor_data(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded Fama-French factor data.

    The factors are:

    - Mom - Momentum
    - Mkt-RF - Market return minus risk-free rate
    - SMB - Small minus big
    - HML - High minus low
    - RMW - Robust minus weak
    - CMA - Conservative minus aggressive
    - LT_Rev - Long-term reversal
    - ST_Rev - Short-term reversal

    The factor data has units of percent per day.

    Args:
        start_date (str, optional): Start date. Defaults to "1995-01-01".
        end_date (str, optional): End date. Defaults to "2025-01-01".

    Returns:
        pd.DataFrame: Factor data with columns for each factor and index for dates.
    """
    factor_dir = Path("data/factors")
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "LT_Rev", "ST_Rev", "Min_Vol"]

    factor_dfs = []
    for file in factor_dir.glob("*.csv"):
        with Path(file).open("r") as f:
            content = f.read()

        # extract csv content between empty lines
        csv_data = content.split("\n\n")[-2]
        buffer = io.StringIO(csv_data)
        factor_returns = pd.read_csv(buffer, parse_dates=[0], index_col=0) / 100.0
        factor_dfs.append(factor_returns)

    return pd.concat(factor_dfs, axis=1).loc[start_date:end_date, factors].sort_index().dropna()


def load_fed_funds_rate(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded Fed Funds Rate data.

    The Fed Funds Rate data has units of multiplier per day.

    Args:
        start_date (str, optional): Start date. Defaults to "1995-01-01".
        end_date (str, optional): End date. Defaults to "2025-01-01".

    Returns:
        pd.DataFrame: Fed Funds Rate data with index for dates.
    """
    ffr_path = Path("data/raw/ffr.parquet")
    return pd.read_parquet(ffr_path).loc[start_date:end_date]


def load_inflation_data(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded inflation data.

    The inflation data has units of multiplier per day.

    Returns:
        pd.DataFrame: Inflation data with index for dates.
    """
    inflation_path = Path("data/raw/cpi.parquet")
    return pd.read_parquet(inflation_path).loc[start_date:end_date]


def load_asset_close_prices(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded asset close prices.

    The asset close prices have units of dollars per share.

    Returns:
        pd.DataFrame: Asset close prices with index for dates.
    """
    closes_path = Path("data/raw/closes.parquet")
    return pd.read_parquet(closes_path).loc[start_date:end_date]


def load_asset_returns(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded asset returns.

    The asset returns have units of percent per day.

    Returns:
        pd.DataFrame: Asset returns with index for dates.
    """
    return load_asset_close_prices(start_date, end_date).pct_change().dropna(how="all")


def load_asset_volumes(start_date: str = "1995-01-01", end_date: str = "2025-01-01") -> pd.DataFrame:
    """Load downloaded volume data.

    The volume data has units of shares per day.

    Returns:
        pd.DataFrame: Volume data with index for dates.
    """
    volumes_path = Path("data/raw/volumes.parquet")
    return pd.read_parquet(volumes_path).loc[start_date:end_date].dropna(how="all")
