# load fred api key to os env
source scripts/env.zsh

uv run python scripts/download.py

uv run python scripts/alphasscri.py

uv run python scripts/universes.py
