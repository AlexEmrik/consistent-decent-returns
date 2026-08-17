import pandas as pd
from pathlib import Path

processed_asset_universes_dir = Path("data/processed/asset_universes")
processed_asset_universes_dir.mkdir(parents=True, exist_ok=True)

closes = pd.read_parquet("./data/raw/closes.parquet")

full_universe = closes.copy()
full_universe[:] = 1.0

full_universe.to_parquet(processed_asset_universes_dir / "full_universe.parquet")

print("Full universe created.")
