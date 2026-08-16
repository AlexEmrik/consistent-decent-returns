"""Performs portfolio backtesting."""

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import decent_returns.backtest.dataloader as dl
from decent_returns.backtest.optimizer import construct_portfolio
from decent_returns.utils import load_from_pkl

from decent_returns.alpha.modules.features import drawdown


def save_bt_results(
    navs: pd.Series,
    composition: pd.DataFrame,
    turnover: pd.Series,
    metadata: pd.Series,
    backtest_name: str,
) -> None:
    """Save backtest results to disk.

    Args:
        navs (pd.Series): Net asset values over time.
        composition (pd.DataFrame): Portfolio composition over time.
        turnover (pd.Series): Turnover over time.
        metadata (pd.Series): Metadata for the backtest.
        backtest_name (str): Name of the backtest.
    """
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / f"{backtest_name}.pkl"
    with path.open("wb") as f:
        pickle.dump({"navs": navs, "composition": composition, "turnover": turnover, "metadata": metadata}, f)


def load_bt_results(backtest_name: str) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Load backtest results from disk.

    Args:
        backtest_name (str): Name of the backtest.

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.Series]: navs, composition, turnover, metadata
    """
    path = Path("results") / f"{backtest_name}.pkl"
    with path.open("rb") as f:
        data = pickle.load(f)  # noqa: S301
        navs = data["navs"]
        composition = data["composition"]
        turnover = data["turnover"]
        metadata = data["metadata"]

    return navs, composition, turnover, metadata


def load_raw_backtest_data() -> dict[str, dict]:
    """Load backtest data from disk.

    Returns:
        dict[str, dict]: Backtest data.
    """
    closes = dl.load_asset_close_prices()
    ffr = dl.load_fed_funds_rate()

    proc_data_dir = Path("data/processed")
    alpha_dir = proc_data_dir / "alphas"
    alphas = {}
    for file in alpha_dir.glob("*.parquet"):
        alphas[file.stem] = pd.read_parquet(file)

    asset_universe_dir = proc_data_dir / "asset_universes"
    asset_universes = {}
    for file in asset_universe_dir.glob("*.parquet"):
        asset_universes[file.stem] = pd.read_parquet(file)

    risk_model_dir = proc_data_dir / "risk_model"
    risk_models = {}
    for file in risk_model_dir.glob("*.pkl"):
        risk_model = load_from_pkl(file)            
        if "Sigmas" in risk_model:
            risk_models[file.stem] = {"Sigmas": risk_model["Sigmas"]}
        else: 
            risk_models[file.stem] = {
                "Fs": risk_model["Fs"],
                "D_halves": risk_model["D_halves"],
        }

    return {
        "closes": closes,
        "ffr": ffr,
        "risk_model": risk_models,
        "asset_universes": asset_universes,
        "alphas": alphas,
    }


def pack_bt_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    alpha: str,
    universe_tickers: list,
    risk_model: str,
    portfolio_closes: str,
    raw_data: dict[str, dict],
) -> dict:
    """Pack raw backtest data into numpy arrays ready for simulation.

    All derived quantities (filled returns, universe masking) are computed here
    so that run_backtest operates entirely on pre-built numpy arrays.

    Args:
        start_date (pd.Timestamp): Start date of the backtest.
        end_date (pd.Timestamp): End date of the backtest.
        alpha (str): Name of the alpha file.
        universe_tickers: Tickers active in this backtest universe.
        universe (str): Name of the universe file.
        risk_model (str): Name of the risk model file.
        raw_data (dict[str, dict]): Raw backtest data.

    Returns:
        dict: Packed backtest data with keys:
            timeline (pd.DatetimeIndex), assets (np.ndarray),
            filled_returns, ffr, universes, alphas,
            Fs, D_halves — all np.ndarray.
    """
    ffr = raw_data["ffr"].loc[start_date:end_date]
    ffr_np = ffr.to_numpy().flatten()
    ffr_np[0] = 0.0  # FFR: zero on the first day (no carry before the simulation starts)

    alphas_np = np.nan_to_num(raw_data["alphas"][alpha].loc[start_date:end_date].to_numpy())

    if portfolio_closes is None:
        closes: pd.DataFrame = raw_data["closes"].loc[start_date:end_date]
        asset_mask = raw_data["closes"].notna().shift(126, fill_value=False).loc[start_date:end_date]

        universes_df = closes.copy().astype(bool)
        universes_df[:] = False
        universes_df[universe_tickers] = True
        universes_np = universes_df.to_numpy() & asset_mask.to_numpy()
    else:
        closes = pd.read_parquet(f"./data/processed/{portfolio_closes}.parquet")
        universes_df = pd.DataFrame(True, columns=universe_tickers, index=closes.index)
        universes_np = universes_df.to_numpy() & closes.notna().to_numpy()

    timeline = closes.index  # pd.DatetimeIndex
    times_np = timeline.to_numpy()

    filled_closes = closes.ffill().bfill()  # Filled returns: ffill then bfill closes, compute daily returns
    prev_closes = filled_closes.shift(1)
    filled_returns_np = np.nan_to_num((filled_closes - prev_closes).to_numpy() / prev_closes.to_numpy())

    assets = closes.columns.to_numpy()

    if "Sigmas" in raw_data["risk_model"][risk_model]:
        Sigma_dict = raw_data["risk_model"][risk_model]["Sigmas"]
        Sigma_np = np.stack([Sigma_dict[t].to_numpy() for t in times_np], axis=0)

        return {
            "timeline": timeline,
            "assets": assets,
            "filled_returns": filled_returns_np,
            "ffr": ffr_np,
            "universes": universes_np,
            "universe_tickers": universe_tickers,
            "alphas": alphas_np,
            "Sigmas": Sigma_np,
        }
    
    else:
        Fs_dict = raw_data["risk_model"][risk_model]["Fs"]
        D_halves_dict = raw_data["risk_model"][risk_model]["D_halves"]
        Fs_np = np.stack([Fs_dict[t].to_numpy() for t in times_np], axis=0)
        D_halves_np = np.stack([D_halves_dict[t].to_numpy() for t in times_np], axis=0)

        return {
            "timeline": timeline,
            "assets": assets,
            "filled_returns": filled_returns_np,
            "ffr": ffr_np,
            "universes": universes_np,
            "universe_tickers": universe_tickers,
            "alphas": alphas_np,
            "Fs": Fs_np,
            "D_halves": D_halves_np,
        }


def run_backtest(
    backtest_name: str,
    rebal_freq: Literal["N", "D", "W", "M", "Q", "Y", "MDD"],
    optimizer_params: dict,
    backtest_data: dict,
) -> None:
    """Run a backtest for a given portfolio strategy.

    Args:
        backtest_name (str): Name of the backtest.
        rebal_freq (Literal["N", "D", "W", "M", "Q", "Y"]): Rebalancing frequency.
        optimizer_params (dict): Parameters for the optimizer.
        backtest_data (dict): Packed backtest data from pack_bt_data.
    """
    timeline: pd.DatetimeIndex = backtest_data["timeline"]
    assets: np.ndarray = backtest_data["assets"]
    filled_returns_1p_np: np.ndarray = 1.0 + backtest_data["filled_returns"]
    ffr_np: np.ndarray = backtest_data["ffr"]
    ffr_1p_np: np.ndarray = 1.0 + ffr_np
    universes_np: np.ndarray = backtest_data["universes"]
    alphas_np: np.ndarray = backtest_data["alphas"]
    Sigmas_np = None
    Fs_np = None
    D_halves_np = None
    
    if "Sigmas" in backtest_data:
        Sigmas_np: np.ndarray = backtest_data["Sigmas"]
    else:
        Fs_np: np.ndarray = backtest_data["Fs"]
        D_halves_np: np.ndarray = backtest_data["D_halves"]

    universe_tickers: list = backtest_data["universe_tickers"]  # needed for correctly assigning max_weights
    all_tickers = assets

    max_weights_config: list = optimizer_params.get("max_weights", [])
    max_weights_full = np.ones(len(assets), dtype=float)  # default: no constraint (1.0)

    if max_weights_config:
        universe_tickers_sorted = sorted(universe_tickers)
        max_weights_sorted = np.array(max_weights_config)[np.argsort(universe_tickers)]
        for ticker, w in zip(universe_tickers_sorted, max_weights_sorted, strict=True):
            idx = np.where(all_tickers == ticker)[0]
            if len(idx) > 0:
                max_weights_full[idx[0]] = w

    bid_ask_spread = optimizer_params.get("bid_ask_spread", 5e-4)
    n_time = len(timeline)

    metadata = pd.Series(
        {
            "name": backtest_name,
            "rebal_freq": rebal_freq,
            "method": optimizer_params.get("method", "markowitz70"),
            "start_date": timeline[0],
            "end_date": timeline[-1],
            "rebal_count": None,
        }
    )

    # Pre-allocate output arrays
    navs_np = np.empty(n_time)
    composition_np = np.zeros((n_time, len(assets)))
    turnover_np = np.zeros(n_time, dtype=float)

    # Build a set of integer indices for O(1) rebal-day lookup
    t_series = pd.Series(timeline)
    if rebal_freq == "N" or rebal_freq == "MDD":
        rebal_idx_set = set()
    else: 
        rebal_idx_set = set(t_series[~t_series.dt.to_period(rebal_freq).duplicated(keep="last")].index.tolist())

    any_liquidation_np: np.ndarray = ~np.all(universes_np, axis=1)

    cash = 1.0
    holdings_np = np.zeros(len(assets))
    weights_np = np.zeros(len(assets))  # reused buffer — never reassigned
    first_solve = True

    mdd_reached = False
    new_asset = False
    # mdd_limit = 0.05  # FIXME
    rebal_count = -1

    prices_np = filled_returns_1p_np.cumprod(axis=0)  # (n_time, n_assets)
    asset_dd_limit = optimizer_params.get("asset_dd_limit", None)

    for i in range(n_time):
        universe_np = universes_np[i]

        cash *= ffr_1p_np[i]
        holdings_np *= filled_returns_1p_np[i]
        turnover_dollars = 0.0
        if any_liquidation_np[i]:
            prev_holdings_sum = holdings_np.sum()
            holdings_np *= universe_np
            liq_value = prev_holdings_sum - holdings_np.sum()
            cash += liq_value
            turnover_dollars = liq_value

        nav = cash + holdings_np.sum() - bid_ask_spread * turnover_dollars
        np.divide(holdings_np, nav, out=weights_np)
        navs_np[i] = nav
        turnover = turnover_dollars / nav
        
        # NOTE
        # nav_max = navs_np[max(0, i-20):i+1].max()
        # mdd = 1.0 - nav / nav_max
        # if mdd > mdd_limit:
        #     mdd_reached = True

        if rebal_freq == "MDD":
            window = prices_np[max(0, i-20):i+1]
            asset_dd = 1.0 - window[-1] / window.max(axis=0)
            if (asset_dd * universe_np > asset_dd_limit).any():
                mdd_reached = True

        if rebal_freq == "N":
            # rebal to add new asset to portfolio, needed for buy and hold strategy (rebal = N)
            new_asset = universes_np[max(0, i-1)].sum() < universes_np[i].sum()

        if i in rebal_idx_set or first_solve or mdd_reached or new_asset:
            rebal_count += 1
            mdd_reached = False
            prev_cash_weight = 1.0 - weights_np.sum()

            F_i = Fs_np[i] if Fs_np is not None else None
            D_half_i = D_halves_np[i] if D_halves_np is not None else None
            Sigma_i = Sigmas_np[i] if Sigmas_np is not None else None

            new_weights = construct_portfolio(
                alphas=alphas_np[i],
                curr_weights=weights_np,
                F=F_i,
                D_half=D_half_i,
                Sigma=Sigma_i,
                universe=universe_np,
                max_weights=max_weights_full,
                ffr=ffr_np[i],
                first_solve=first_solve,
                **{k: v for k, v in optimizer_params.items() if k != "max_weights"},
            )
            first_solve = False

            new_cash_weight = 1.0 - new_weights.sum()
            tcost_base = np.abs(new_weights - weights_np).sum()
            turnover += 0.5 * (tcost_base)
            new_cash_weight -= 0.5 * bid_ask_spread * tcost_base

            cash = new_cash_weight * nav
            np.multiply(new_weights, nav, out=holdings_np)
            weights_np[:] = new_weights

        composition_np[i] = weights_np
        turnover_np[i] = turnover

    composition = pd.DataFrame(composition_np, index=timeline, columns=assets)
    composition["Cash"] = 1.0 - composition.sum(axis=1)
    navs = pd.Series(navs_np, index=timeline)
    turnover = pd.Series(turnover_np, index=timeline)
    metadata["rebal_count"] = rebal_count
    save_bt_results(navs, composition, turnover, metadata, backtest_name)
