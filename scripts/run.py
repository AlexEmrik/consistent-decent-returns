"""Run portfolio backtests."""

import argparse
import json
import sys
import warnings
from multiprocessing import set_start_method
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from decent_returns.backtest.simulator import load_raw_backtest_data, pack_bt_data, run_backtest

set_start_method("fork", force=True)
warnings.filterwarnings("ignore", category=UserWarning)  # cvxpy inaccuracies

MAX_PROCESSES = 12

START_DATE = pd.Timestamp("2005-01-01") # pd.Timestamp("2005-01-01")  # pd.Timestamp("2010-01-01")
END_DATE =   pd.Timestamp("2025-01-01") # pd.Timestamp("2020-01-01")  # pd.Timestamp("2025-01-01")

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run portfolio backtests.")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument(
    "--all",
    action="store_true",
    help="Run all backtests from configs/*.json",
)
mode.add_argument(
    "-f",
    metavar="CONFIG_FILE",
    help="Run all backtests defined in a specific JSON config file",
)
args = parser.parse_args()

# ── Resolve config glob ────────────────────────────────────────────────────────

config_dir = Path("configs/backtest")
config_files = list(config_dir.glob("*.json")) if args.all else [Path(args.f)]

if not config_files:
    print("No config files matched. Nothing to run.")
    sys.exit(0)

param_sets: dict = {}
for file in config_files:
    with file.open("r") as f:
        param_sets = param_sets | json.load(f)

if not param_sets:
    print("No backtest configs found. Nothing to run.")
    sys.exit(0)

# ── Load data and build batches ────────────────────────────────────────────────

print("Loading backtest data...")
all_backtest_data = load_raw_backtest_data()

param_batches = []
for name, params in param_sets.items():
    rebal_freq = params.pop("rebal_freq")
    backtest_data = pack_bt_data(
        start_date=START_DATE,
        end_date=END_DATE,
        alpha=params.pop("alphas"),
        universe_tickers=params.pop("asset_universes"),
        risk_model=params.pop("risk_model"),
        portfolio_closes=params.pop("closes") if "closes" in params else None,
        raw_data=all_backtest_data,
    )
    optimizer_params = params.pop("optimizer_params", {})
    if "target_vol" in optimizer_params:
        optimizer_params["target_vol"] = optimizer_params["target_vol"] / np.sqrt(252)
    param_batches.append((name, rebal_freq, optimizer_params, backtest_data))


# ── Runner ─────────────────────────────────────────────────────────────────────


def bt_wrapper(name: str, rebal_freq: str, optimizer_params: dict, backtest_data: dict) -> None:
    """Wrapper to run a single backtest."""
    run_backtest(name, rebal_freq, optimizer_params, backtest_data)


print(f"Running {len(param_batches)} backtests...")
names, rebal_freqs, optimizer_params_list, backtest_data_list = zip(*param_batches, strict=True)
process_map(
    bt_wrapper,
    names,
    rebal_freqs,
    optimizer_params_list,
    backtest_data_list,
    max_workers=MAX_PROCESSES,
    desc="Backtests",
)
