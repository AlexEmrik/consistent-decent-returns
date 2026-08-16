# consistent_decent_returns

Long only portfolio optimization to achieve consistent, decent returns via convex optimization.

---

## Some notes

The ETF data is downloaded from Yahoo Finance and the risk free rate from FRED, using `scripts/download.py`.

`data/`, `results/` and `keys/` are gitignored, so a fresh clone contains no data. Run the download step yourself.

## Core workflow

0. If you don't have uv installed, run `make uv` to install it. We use `uv` to manage the Python environment.
1. Run `make setup` to set up the Python environment and install the pre-commit hooks (keeps code quality high).
2. Add your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) to `keys/fred_api.key`:
   ```bash
   echo YOUR_KEY > keys/fred_api.key
   ```
3. Run `source scripts/env.zsh` to export `FRED_API_KEY` into your shell.
4. Run `uv run python scripts/download.py` once to download the raw data and compute the factor risk model.
5. Run `uv run python scripts/alphas.py` and `uv run python scripts/universes.py` to build the alpha signals and asset universes.
6. Define backtests as JSON config files, then run them with `uv run python scripts/run.py`. Use `--all` for everything, or `-f <file>` for a single config.

Steps 3-5 are wrapped by `zsh scripts/setup.zsh`.

As is, there are two separate but related projects within this repository:

1. Identifying simple portfolio strategies that yield greater returns than a 60/40 portfolio (with small drawdowns).
2. Using machine learning methods to generate alphas for use in a modern Markowitz-at-seventy portfolio construction method.

---

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
│   ├── alphas.py        # Generate alpha signal files
│   ├── universes.py     # Generate asset universe files
│   ├── run.py           # Run backtests (--all / -f)
│   ├── env.zsh          # Export FRED_API_KEY from keys/fred_api.key
│   └── setup.zsh        # End to end data setup
└── data/                # Data, processed and raw
```

## Makefile commands

```bash
make uv              # Install uv
make setup           # Setup the project venv and install pre-commit hooks
make fmt             # Run autoformatting and linting
```