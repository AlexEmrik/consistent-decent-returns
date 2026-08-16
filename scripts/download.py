"""Download raw data and compute shared processed artifacts."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm
import os

import decent_returns.backtest.dataloader as dl
from decent_returns.risk.stat import get_F_D_half
from decent_returns.utils import save_to_pkl

from decent_returns.config import universe, cryp
from decent_returns.config import START_DATE, END_DATE

raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

print("=== Stage 1: Downloading raw data ===")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Parameters ────────────────────────────────────────────────────────────────

VOL_HALFLIFE = 42           # FIXME revert back to 42
COV_HALFLIFE = 126          # FIXME revert back to 126
UNIVERSE_MIN_PERIODS = 126  # FIXME revert back to 126
MIN_UNIVERSE_SIZE = 15      # FIXME revert back to 15
N_FACTORS = 10              # FIXME revert back to 10
FIRST_DATE_CRYPTO =  "2017-09-07"
etfs = universe

# ── Stage 1: Download raw data ─────────────────────────────────────────────────

yf_data = yf.download(etfs, start=START_DATE, end=END_DATE, auto_adjust=True)
closes = yf_data["Close"]
volumes = yf_data["Volume"]

if "BTC-USD" in universe:
    closes.rename( {"ETH-USD": "ETHA", "BTC-USD":"IBIT"})
    closes.loc[:FIRST_DATE_CRYPTO,  "BTC-USD"] = np.nan
    volumes.loc[:FIRST_DATE_CRYPTO, "BTC-USD"] = np.nan

if "ETH-USD" in universe:
    volumes.rename({"ETH-USD": "ETHA", "BTC-USD":"IBIT"})
    closes.loc[:FIRST_DATE_CRYPTO,  "ETH-USD"] = np.nan
    volumes.loc[:FIRST_DATE_CRYPTO, "ETH-USD"] = np.nan

# exclude crypto from trading days since they trade on weekends as well
trading_days = closes.drop(columns=cryp).dropna(how="all").index
closes = closes.loc[trading_days]
volumes = volumes.loc[trading_days]
closes.to_parquet(raw_data_dir / "closes.parquet")
volumes.to_parquet(raw_data_dir / "volumes.parquet")

print("Saved closes.parquet and volumes.parquet")

print("Downloading VIX, yield curve, and DXY data...")
special_tickers = ["^VIX", "^FVX", "^TYX", "DX-Y.NYB"]
rename_map = {"^VIX": "VIX", "^FVX": "US5Y", "^TYX": "US30Y", "DX-Y.NYB": "DXY"}
special_data = yf.download(special_tickers, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
special_data = special_data.rename(columns=rename_map)
special_data[["VIX"]].to_parquet(raw_data_dir / "vix.parquet")
yield_data = special_data[["US5Y", "US30Y"]].copy()
yield_data["Slope"] = (yield_data["US30Y"] - yield_data["US5Y"]) / 25.0
yield_data["Intercept"] = yield_data["US5Y"] - (yield_data["Slope"] * 5.0)
yield_data = yield_data.drop(columns=["US5Y", "US30Y"])
yield_data.to_parquet(raw_data_dir / "yield_curve.parquet")
special_data[["DXY"]].to_parquet(raw_data_dir / "dxy.parquet")
print("Saved vix.parquet, yield_curve.parquet, dxy.parquet")

FRED_API_KEY = os.environ.get("FRED_API_KEY")
print(FRED_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

print("Downloading Fed Funds Rate...")
ffr_raw = fred.get_series("DFF", observation_start=START_DATE, observation_end=END_DATE).to_frame("FFR")
ffr_aligned = ffr_raw.reindex(yf_data.index).ffill()
ffr_daily = (1 + ffr_aligned / 100) ** (1 / 252) - 1.0
ffr_daily.to_parquet(raw_data_dir / "ffr.parquet")
print("Saved ffr.parquet")

cpi_raw = fred.get_series("CPILFESL", observation_start=START_DATE, observation_end=END_DATE).to_frame("CPI")
union_index = yf_data.index.union(cpi_raw.index)
cpi_daily = (1 + cpi_raw.pct_change()) ** (1 / 21)
cpi_daily = cpi_daily.reindex(union_index).ffill().reindex(yf_data.index)
cpi_daily.to_parquet(raw_data_dir / "cpi.parquet")
print("Saved cpi.parquet")

print("Downloading Unemployment Rate...")
unr_raw = fred.get_series("UNRATE", observation_start=START_DATE, observation_end=END_DATE).to_frame("UNR")
union_index = yf_data.index.union(unr_raw.index)
unr_aligned = unr_raw.reindex(union_index).ffill().reindex(yf_data.index)
unr_aligned.to_parquet(raw_data_dir / "unr.parquet")
print("Saved unr.parquet")

print("Downloading Real GDP per capita...")
gdp_raw = fred.get_series("A939RX0Q048SBEA", observation_start=START_DATE, observation_end=END_DATE).to_frame(
    "GDP"
)
union_index = yf_data.index.union(gdp_raw.index)
gdp_growth = gdp_raw.reindex(union_index).ffill().reindex(yf_data.index).pct_change()
gdp_growth.to_parquet(raw_data_dir / "gdp.parquet")
print("Saved gdp.parquet")

print("Downloading PCE...")
pce_raw = fred.get_series("DPCCRV1Q225SBEA", observation_start=START_DATE, observation_end=END_DATE).to_frame(
    "PCE"
)
union_index = yf_data.index.union(pce_raw.index)
pce_aligned = pce_raw.reindex(union_index).ffill().reindex(yf_data.index)
pce_aligned.to_parquet(raw_data_dir / "pce.parquet")
print("Saved pce.parquet")

print("Downloading HY corporate yield...")
hyy_raw = fred.get_series("BAMLH0A0HYM2EY", observation_start=START_DATE, observation_end=END_DATE).to_frame("HYY")
hyy_aligned = hyy_raw.reindex(yf_data.index).ffill()
hyy_aligned.to_parquet(raw_data_dir / "hyy.parquet")
print("Saved hyy.parquet")

print("Downloading 10Y Treasury yield...")
dgs10_raw = fred.get_series("DGS10", observation_start=START_DATE, observation_end=END_DATE).to_frame("DGS10")
dgs10_aligned = dgs10_raw.reindex(yf_data.index).ffill()
dgs10_aligned.to_parquet(raw_data_dir / "dgs10.parquet")
print("Saved dgs10.parquet")

print("Downloading 10Y Treasury yield...")
dgs10_raw = fred.get_series("DGS10", observation_start=START_DATE, observation_end=END_DATE).to_frame("DGS10")
dgs10_aligned = dgs10_raw.reindex(yf_data.index).ffill()
dgs10_aligned.to_parquet(raw_data_dir / "dgs10.parquet")
print("Saved dgs10.parquet")

print("Downloading 30Y Treasury yield...")
dgs30_raw = fred.get_series("DGS30", observation_start=START_DATE, observation_end=END_DATE).to_frame("DGS30")
dgs30_aligned = dgs30_raw.reindex(yf_data.index).ffill()
dgs30_aligned.to_parquet(raw_data_dir / "dgs30.parquet")
print("Saved dgs30.parquet")

print("Downloading 10Y TIPS yield...")
dfii10_raw = fred.get_series("DFII10", observation_start=START_DATE, observation_end=END_DATE).to_frame("DFII10")
dfii10_aligned = dfii10_raw.reindex(yf_data.index).ffill()
dfii10_aligned.to_parquet(raw_data_dir / "dfii10.parquet")
print("Saved dfii10.parquet")

print("Downloading IG corporate yield...")
igy_raw = fred.get_series("BAMLC0A0CMEY", observation_start=START_DATE, observation_end=END_DATE).to_frame("IGY")
igy_aligned = igy_raw.reindex(yf_data.index).ffill()
igy_aligned.to_parquet(raw_data_dir / "igy.parquet")
print("Saved igy.parquet")

print("Downloading HY corporate yield...")
hyy_raw = fred.get_series("BAMLH0A0HYM2EY", observation_start=START_DATE, observation_end=END_DATE).to_frame("HYY")
hyy_aligned = hyy_raw.reindex(yf_data.index).ffill()
hyy_aligned.to_parquet(raw_data_dir / "hyy.parquet")
print("Saved hyy.parquet")

print("Downloading EM corporate yield...")
emy_raw = fred.get_series("BAMLEMHBHYCRPIEY", observation_start=START_DATE, observation_end=END_DATE).to_frame("EMY")
emy_aligned = emy_raw.reindex(yf_data.index).ffill()
emy_aligned.to_parquet(raw_data_dir / "emy.parquet")
print("Saved emy.parquet")

# commodity spot prices
print("Downloading gold spot...")
gc_raw = yf.download("GC=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("GC")
gc_aligned = gc_raw.reindex(yf_data.index).ffill()
gc_aligned.to_parquet(raw_data_dir / "gc.parquet")
print("Saved gc.parquet")

print("Downloading silver spot...")
si_raw = yf.download("SI=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("SI")
si_aligned = si_raw.reindex(yf_data.index).ffill()
si_aligned.to_parquet(raw_data_dir / "si.parquet")
print("Saved si.parquet")

print("Downloading copper spot...")
hg_raw = yf.download("HG=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("HG")
hg_aligned = hg_raw.reindex(yf_data.index).ffill()
hg_aligned.to_parquet(raw_data_dir / "hg.parquet")
print("Saved hg.parquet")

print("Downloading WTI crude spot...")
cl_raw = yf.download("CL=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("CL")
cl_aligned = cl_raw.reindex(yf_data.index).ffill()
cl_aligned.to_parquet(raw_data_dir / "cl.parquet")
print("Saved cl.parquet")

print("Downloading gasoline spot...")
rb_raw = yf.download("RB=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("RB")
rb_aligned = rb_raw.reindex(yf_data.index).ffill()
rb_aligned.to_parquet(raw_data_dir / "rb.parquet")
print("Saved rb.parquet")

print("Downloading corn spot...")
zc_raw = yf.download("ZC=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("ZC")
zc_aligned = zc_raw.reindex(yf_data.index).ffill()
zc_aligned.to_parquet(raw_data_dir / "zc.parquet")
print("Saved zc.parquet")

print("Downloading wheat spot...")
zw_raw = yf.download("ZW=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("ZW")
zw_aligned = zw_raw.reindex(yf_data.index).ffill()
zw_aligned.to_parquet(raw_data_dir / "zw.parquet")
print("Saved zw.parquet")

print("Downloading soybeans spot...")
zs_raw = yf.download("ZS=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("ZS")
zs_aligned = zs_raw.reindex(yf_data.index).ffill()
zs_aligned.to_parquet(raw_data_dir / "zs.parquet")
print("Saved zs.parquet")

print("Downloading sugar spot...")
sb_raw = yf.download("SB=F", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"].squeeze().to_frame("SB")
sb_aligned = sb_raw.reindex(yf_data.index).ffill()
sb_aligned.to_parquet(raw_data_dir / "sb.parquet")
print("Saved sb.parquet")

# ── Stage 2: Full-universe risk model ─────────────────────────────────────────

# print("\n=== Stage 2: Building full-universe risk model ===")

proc_data_dir = Path("data/processed")
proc_data_dir.mkdir(parents=True, exist_ok=True)

returns = dl.load_asset_returns()
assets = returns.columns

factors = list(range(N_FACTORS))

Fs: dict = {}
D_halves: dict = {}

for ts in tqdm(returns.index[UNIVERSE_MIN_PERIODS:], desc="Computing Fs and D_halves"):
    returns_so_far = returns.loc[:ts].copy()
    universe = returns_so_far.tail(UNIVERSE_MIN_PERIODS).dropna(axis=1).columns
    if len(universe) < MIN_UNIVERSE_SIZE:
        Fs[ts] = pd.DataFrame(0.0, index=assets, columns=factors)
        D_halves[ts] = pd.Series(0.0, index=assets)
        continue
    returns_so_far = returns_so_far[universe].dropna()
    F, D_half = get_F_D_half(
        returns_so_far,
        k=N_FACTORS,
        vol_halflife=VOL_HALFLIFE,
        cov_halflife=COV_HALFLIFE,
    )
    Fs[ts] = F.reindex(index=assets, columns=factors, fill_value=0.0)
    D_halves[ts] = D_half.reindex(index=assets, fill_value=0.0)

risk_model_dir = proc_data_dir / "risk_model"
risk_model_dir.mkdir(parents=True, exist_ok=True)
save_to_pkl({"Fs": Fs, "D_halves": D_halves}, risk_model_dir / f"K{N_FACTORS}.pkl")
print(f"Saved K{N_FACTORS}.pkl")

# ── Stage 3: Base asset universes and uncertainties ────────────────────────────

# print("\n=== Stage 3: Building  uncertainties ===")

# no_uncertainty = pd.DataFrame(index=returns.index, columns=returns.columns, data=0.0, dtype=float)

# uncertainties_dir = proc_data_dir / "uncertainties"
# uncertainties_dir.mkdir(parents=True, exist_ok=True)
# no_uncertainty.to_parquet(uncertainties_dir / "no_uncertainty.parquet")

# print("Saved no_uncertainty.parquet")

# # ── Stage 4: Extract Macro and Return features ─────────────────────────────────


# print("\n=== Stage 4: Building return and macro features ===")

# ffr_aligned = ffr_daily.reindex(returns.index)
# rx = returns.sub(ffr_aligned["FFR"], axis=0)
# rx_log = np.log(rx + 1)

# _2W = 10
# _1M = 21
# _3M = 63
# _6M = 126
# _1Y = 252
# _2Y = 504
# _3Y = 756

# mom_2w = rx_log.rolling(_2W).sum()
# mom_1m = rx_log.rolling(_1M).sum()
# mom_3m = rx_log.rolling(_3M).sum()
# mom_6m = rx_log.rolling(_6M).sum()
# mom_1y = rx_log.rolling(_1Y).sum()
# mom_2y = rx_log.rolling(_2Y).sum()
# mom_3y = rx_log.rolling(_3Y).sum()

# vol_2w = rx.rolling(_2W).std()
# vol_1m = rx.rolling(_1M).std()
# vol_3m = rx.rolling(_3M).std()
# vol_6m = rx.rolling(_6M).std()
# vol_1y = rx.rolling(_1Y).std()
# vol_2y = rx.rolling(_2Y).std()
# vol_3y = rx.rolling(_3Y).std()

# mvs_2w = mom_2w / (vol_2w * np.sqrt(_2W))
# mvs_1m = mom_1m / (vol_1m * np.sqrt(_1M))
# mvs_3m = mom_3m / (vol_3m * np.sqrt(_3M))
# mvs_6m = mom_6m / (vol_6m * np.sqrt(_6M))
# mvs_1y = mom_1y / (vol_1y * np.sqrt(_1Y))
# mvs_2y = mom_2y / (vol_2y * np.sqrt(_2Y))
# mvs_3y = mom_3y / (vol_3y * np.sqrt(_3Y))


# def max_drawdown(x):
#     running_max = x.cummax()
#     dd = (x - running_max) / running_max
#     return -dd.min()


# mdd_2w = closes.rolling(_2W).apply(max_drawdown)
# mdd_1m = closes.rolling(_1M).apply(max_drawdown)
# mdd_3m = closes.rolling(_3M).apply(max_drawdown)
# mdd_6m = closes.rolling(_6M).apply(max_drawdown)
# mdd_1y = closes.rolling(_1Y).apply(max_drawdown)
# mdd_2y = closes.rolling(_2Y).apply(max_drawdown)
# mdd_3y = closes.rolling(_3Y).apply(max_drawdown)

# feature_frames = {
#     "mom_2w": mom_2w,
#     "mom_1m": mom_1m,
#     "mom_3m": mom_3m,
#     "mom_6m": mom_6m,
#     "mom_1y": mom_1y,
#     "mom_2y": mom_2y,
#     "mom_3y": mom_3y,
#     "vol_2w": vol_2w,
#     "vol_1m": vol_1m,
#     "vol_3m": vol_3m,
#     "vol_6m": vol_6m,
#     "vol_1y": vol_1y,
#     "vol_2y": vol_2y,
#     "vol_3y": vol_3y,
#     "mvs_2w": mvs_2w,
#     "mvs_1m": mvs_1m,
#     "mvs_3m": mvs_3m,
#     "mvs_6m": mvs_6m,
#     "mvs_1y": mvs_1y,
#     "mvs_2y": mvs_2y,
#     "mvs_3y": mvs_3y,
#     "mdd_2w": mdd_2w,
#     "mdd_1m": mdd_1m,
#     "mdd_3m": mdd_3m,
#     "mdd_6m": mdd_6m,
#     "mdd_1y": mdd_1y,
#     "mdd_2y": mdd_2y,
#     "mdd_3y": mdd_3y,
# }

# asset_features: pd.DataFrame = pd.concat(feature_frames, keys=feature_frames.keys(), axis=1)
# asset_features = asset_features.swaplevel(0, 1, axis=1).sort_index(axis=1)
# asset_features.columns.names = ["asset", "feature"]

# features_dir = proc_data_dir / "features"
# features_dir.mkdir(parents=True, exist_ok=True)
# asset_features.to_parquet(features_dir / "asset.parquet")

# vix       = pd.read_parquet(raw_data_dir / "vix.parquet").reindex(returns.index).ffill()
# hyy       = pd.read_parquet(raw_data_dir / "hyy.parquet").reindex(returns.index).ffill()
# dgs10     = pd.read_parquet(raw_data_dir / "dgs10.parquet").reindex(returns.index).ffill()
# dxy       = pd.read_parquet(raw_data_dir / "dxy.parquet").reindex(returns.index).ffill()
# dgs30     = pd.read_parquet(raw_data_dir / "dgs30.parquet").reindex(returns.index).ffill()
# dfii10    = pd.read_parquet(raw_data_dir / "dfii10.parquet").reindex(returns.index).ffill()
# igy       = pd.read_parquet(raw_data_dir / "igy.parquet").reindex(returns.index).ffill()
# emy       = pd.read_parquet(raw_data_dir / "emy.parquet").reindex(returns.index).ffill()

# ffr_lvl   = ffr_raw.reindex(returns.index).ffill()["FFR"]

# ffr_d1m   = ffr_lvl.diff(_1M)
# ffr_d1y   = ffr_lvl.diff(_1Y)

# slp_lvl   = yield_data["Slope"].reindex(returns.index).ffill()
# slp_d1m   = slp_lvl.diff(_1M)

# cs_lvl    = (hyy["HYY"] - dgs10["DGS10"])
# cs_d1m    = cs_lvl.diff(_1M)

# vix_lvl   = vix["VIX"]
# vix_d1m   = vix_lvl.diff(_1M)

# cpi_lvl   = cpi_raw.reindex(returns.index).ffill()["CPI"]
# cpi_yoy   = cpi_lvl / cpi_lvl.shift(_1Y) - 1.0

# spy_12m   = rx_log["SPY"].rolling(_1Y).sum() if "SPY" in rx_log.columns else \
#             np.log(closes["SPY"] / closes["SPY"].shift(_1Y))

# dgs10_d1m = dgs10["DGS10"].diff(_1M)
# dxy_d1m   = dxy["DXY"].pct_change(_1M)

# dfii10_d1m = dfii10["DFII10"].diff(_1M)
# dxy_d3m    = dxy["DXY"].pct_change(_3M)

# # ── Bond carry yields ───────────────────
# dgs10_lvl  = dgs10["DGS10"]
# dgs30_lvl  = dgs30["DGS30"]
# dfii10_lvl = dfii10["DFII10"]
# igy_lvl    = igy["IGY"]
# hyy_lvl    = hyy["HYY"]
# emy_lvl    = emy["EMY"]

# # ── Commodity basis (spot vs. ETF, Gorton-Rouwenhorst style) ───────────────

# gc = pd.read_parquet(raw_data_dir / "gc.parquet").reindex(returns.index).ffill()
# si = pd.read_parquet(raw_data_dir / "si.parquet").reindex(returns.index).ffill()
# hg = pd.read_parquet(raw_data_dir / "hg.parquet").reindex(returns.index).ffill()
# cl = pd.read_parquet(raw_data_dir / "cl.parquet").reindex(returns.index).ffill()
# rb = pd.read_parquet(raw_data_dir / "rb.parquet").reindex(returns.index).ffill()
# zc = pd.read_parquet(raw_data_dir / "zc.parquet").reindex(returns.index).ffill()
# zw = pd.read_parquet(raw_data_dir / "zw.parquet").reindex(returns.index).ffill()
# zs = pd.read_parquet(raw_data_dir / "zs.parquet").reindex(returns.index).ffill()
# sb = pd.read_parquet(raw_data_dir / "sb.parquet").reindex(returns.index).ffill()

# spot_map = {
#     "gld":  gc["GC"],
#     "slv":  si["SI"],
#     "cper": hg["HG"],
#     "uso":  cl["CL"],
#     "uga":  rb["RB"],
#     "corn": zc["ZC"],
#     "weat": zw["ZW"],
#     "soyb": zs["ZS"],
#     "cane": sb["SB"],
# }

# basis_feats = {}
# for etf, spot in spot_map.items():
#     spot_log = np.log(spot)
#     basis_feats[f"{etf}_basis_3m"] = spot_log.diff(_3M) - rx_log[etf.upper()].rolling(_3M).sum()
#     basis_feats[f"{etf}_basis_1y"] = spot_log.diff(_1Y) - rx_log[etf.upper()].rolling(_1Y).sum()

# shared_macro = pd.DataFrame({
#     "ffr_d1m":  ffr_d1m,
#     "ffr_d1y":  ffr_d1y,
#     "slp_lvl":  slp_lvl,
#     "slp_d1m":  slp_d1m,
#     "cs_lvl":   cs_lvl,
#     "cs_d1m":   cs_d1m,
#     "vix_lvl":  vix_lvl,
#     "vix_d1m":  vix_d1m,
#     "cpi_yoy":  cpi_yoy,
#     "dgs10_d1m": dgs10_d1m,
#     "dxy_d1m": dxy_d1m,
#     "spy_12m":  spy_12m,
#     "dgs10_lvl":  dgs10_lvl,
#     "dgs30_lvl":  dgs30_lvl,
#     "dfii10_lvl": dfii10_lvl,
#     "dfii10_d1m": dfii10_d1m,
#     "dxy_d3m":    dxy_d3m,
#     "igy_lvl":    igy_lvl,
#     "hyy_lvl":    hyy_lvl,
#     "emy_lvl":    emy_lvl,
#     **basis_feats,
# })

# shared_macro.to_parquet(features_dir / "macro.parquet")
# print("Saved macro.parquet and asset.parquet")
print("\nSetup complete.")