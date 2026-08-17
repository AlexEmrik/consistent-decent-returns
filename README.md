# Consistent Decent Returns

Long only portfolio optimization to achieve consistent, decent returns via convex optimization.


## Setup instructions

1. Run the following command to install `uv`.
```bash
make uv              # Install uv
make setup           # Setup the project venv
```

2. Create a file to store your FRED API key required to download the data. Get your key at [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).
```bash
mkdir keys
touch keys/fred_api.key
cat > [YOUR KEY HERE]
```

3. Run the setup scripts with the following command. 
```bash
source scripts/setup.zsh
```

5. Define backtests as JSON files under `./configs`, then run them with the following command
```bash
uv run python scripts/run.py -f configs/path/to/config.json
```



## Project Structure

```text
decent_returns
├── src/decent_returns/
│   ├── alpha/           # Alpha signals
│   ├── backtest/        # Dataloader, optimizer, simulator
│   ├── risk/            # Risk models
│   ├── config.py        # Universe definitions, date ranges, constants
│   ├── utils.py         # Shared utilities
│   └── vis.py           # Visualization helpers
├── scripts/
│   ├── download.py      # Download raw data + build the risk model
│   ├── universes.py     # Generate asset universe file
│   ├── mask.py          # Generate asset mask file
│   ├── alphas.py        # Generate alpha files
│   ├── run.py           # Run backtests (--all / -f)
│   ├── env.zsh          # Export FRED_API_KEY from keys/fred_api.key
│   └── setup.zsh        # End to end data setup
└── data/                # Data, processed and raw
```
