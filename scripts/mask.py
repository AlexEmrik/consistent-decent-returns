import pandas as pd
from pathlib import Path

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

closes = pd.read_parquet("./data/raw/closes.parquet")

asset_mask = closes.notna()

asset_mask.to_parquet(processed_dir / "asset_mask.parquet")

print("Asset mask created.")
