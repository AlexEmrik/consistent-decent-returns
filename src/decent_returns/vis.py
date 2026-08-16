import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    # NOTE HACK
    asset_mask = pd.read_parquet("./data/processed/asset_mask.parquet")

    from pathlib import Path

    import plotly.graph_objects as go

    return Path, go, mo, np, pd, asset_mask


@app.cell
def _():
    from decent_returns.backtest.simulator import load_bt_results
    from decent_returns.backtest.dataloader import (
        load_asset_close_prices,
        load_fed_funds_rate,
        load_inflation_data,
    )

    return load_bt_results, load_asset_close_prices, load_fed_funds_rate, load_inflation_data


@app.cell
def _(pd):
    start_date = pd.Timestamp("2000-01-01")
    end_date = pd.Timestamp("2025-01-01")
    return end_date, start_date


@app.cell
def _(
    Path,
    end_date,
    load_bt_results,
    load_asset_close_prices,
    load_fed_funds_rate,
    load_inflation_data,
    start_date,
):
    res_path = Path("results/")
    backtests = [_f.stem for _f in res_path.glob("*.pkl")]
    results = {_bt: load_bt_results(_bt) for _bt in backtests}
    fed_funds_rate = load_fed_funds_rate().loc[start_date:end_date, "FFR"]
    cpi = load_inflation_data().loc[start_date:end_date, "CPI"] - 1.0

    _bench_px = load_asset_close_prices().loc[start_date:end_date, ["SPY", "AGG", "GLD"]]
    bench_returns = _bench_px.pct_change().dropna()

    return cpi, fed_funds_rate, results, bench_returns


@app.cell
def _(bench_returns, cpi, fed_funds_rate, np, pd, results, asset_mask):
    results_frame = pd.DataFrame(
        columns=[
            "Return",
            "Volatility",
            "MDD",
            "TO",
            "Sharpe Ratio",
            "LQ",
            "Corr SPY",
            "Corr AGG",
            "Corr GLD",
            "Rebal Freq.",
            "Method",
        ]
    )
    results_frame.index.name = "Backtest"

    all_navs = pd.DataFrame()

    for _bt_name, (_navs, _comp, _turn, _meta) in results.items():
        _returns = _navs.pct_change().dropna()
        _cumu_rets = _navs.iloc[-1] / _navs.iloc[0]
        _cumu_ffr = (1 + fed_funds_rate.loc[_returns.index]).prod()

        _mean_returns = _cumu_rets ** (252 / len(_returns)) - 1
        _mean_ffr_returns = (_cumu_rets / _cumu_ffr) ** (252 / len(_returns)) - 1

        _vol = _returns.std() * np.sqrt(252)
        _ffr_sharpe = _mean_ffr_returns / _vol

        _cummax = _navs.cummax()
        _drawdown = (_cummax - _navs) / _cummax

        # --- LQ: 25th percentile of per-calendar-year Sharpe ratios ---
        _ffr_aligned = fed_funds_rate.reindex(_returns.index).fillna(0.0)
        _yearly_sharpes = []
        for _year, _r_y in _returns.groupby(_returns.index.year):
            _cumu_y = (1 + _r_y).prod()
            _cumu_ffr_y = (1 + _ffr_aligned.loc[_r_y.index]).prod()
            _excess_y = (_cumu_y / _cumu_ffr_y) ** (252 / len(_r_y)) - 1
            _vol_y = _r_y.std() * np.sqrt(252)
            _sharpe_y = _excess_y / _vol_y
            if _sharpe_y != 0.0:
                _yearly_sharpes.append(_sharpe_y)
        _lq = np.quantile(_yearly_sharpes, 0.25) if _yearly_sharpes else np.nan

        # Date-aligned correlation vs benchmarks
        _aligned = pd.concat([_returns.rename("p"), bench_returns], axis=1).dropna()
        _corr = _aligned.corr()["p"]

        results_frame.loc[_bt_name] = {
            "Return": _mean_returns,
            "Volatility": _vol,
            "MDD": _drawdown.max(),
            "TO": _turn.mean() * 252,
            "Sharpe Ratio": _ffr_sharpe,
            "LQ": _lq,
            "Corr SPY": _corr["SPY"],
            "Corr AGG": _corr["AGG"],
            "Corr GLD": _corr["GLD"],
            "Rebal Freq.": _meta["rebal_freq"],
            "Method": _meta["method"],
        }
        all_navs[_bt_name] = _navs

    return all_navs, results_frame


@app.cell
def _(mo):
    mo.md(r"""# All Backtests""")
    return


@app.cell
def _(mo, results_frame):
    _fmt_map = {
        "Return": "{:.2%}",
        "Volatility": "{:.2%}",
        "MDD": "{:.2%}",
        "TO": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Corr SPY": "{:.2f}",
        "Corr AGG": "{:.2f}",
        "Corr GLD": "{:.2f}",
    }

    result_selector = mo.ui.table(data=results_frame, selection="multi", format_mapping=_fmt_map)
    result_selector
    return (result_selector,)


@app.cell
def _(go, result_selector, results_frame):
    pareto_fig = go.Figure()

    plt_min = results_frame[["Return", "Volatility"]].min(axis=None) - 0.01
    plt_max = results_frame[["Return", "Volatility"]].max(axis=None) + 0.01

    pareto_fig.add_trace(
        go.Scatter(
            x=result_selector.value["Volatility"],
            y=result_selector.value["Return"],
            mode="markers+text",
            text=result_selector.value.index,
            textposition="bottom center",
            showlegend=False,
            name="",
            marker=dict(
                color=result_selector.value["Sharpe Ratio"],
                colorscale="Inferno",
                showscale=False,
                cmin=0.0,
                cmax=1.2 * results_frame["Sharpe Ratio"].max(),
            ),
        )
    )

    pareto_fig.add_trace(
        go.Scatter(
            x=[plt_min, plt_max],
            y=[plt_min, plt_max],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        )
    )

    pareto_fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
    pareto_fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

    pareto_fig.update_layout(
        title="Pareto Front: Risk vs Return",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
        xaxis=dict(
            range=[plt_min, plt_max],
        ),
        yaxis=dict(
            range=[plt_min, plt_max],
        ),
    )
    return


@app.cell
def _(all_navs, go, np, result_selector):
    nav_comparison = go.Figure()

    _x_min = all_navs.index.min()
    _x_max = all_navs.index.max()
    _y_min = np.log10(all_navs.min(axis=None)) - 0.1
    _y_max = np.log10(all_navs.max(axis=None)) + 0.1

    for _bt_name in result_selector.value.index:
        _navs = all_navs[_bt_name]
        nav_comparison.add_trace(
            go.Scatter(
                x=_navs.index,
                y=_navs,
                mode="lines",
                name=_bt_name,
            )
        )

    nav_comparison.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        range=[_x_min.isoformat(), _x_max.isoformat()],
    )
    nav_comparison.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        range=[_y_min, _y_max],
    )

    nav_comparison.update_yaxes(type="log")
    nav_comparison.update_layout(
        title="Cumulative Return Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
    )
    return


@app.cell
def _(go, result_selector, results):
    comp_fig = go.Figure()

    if len(result_selector.value) > 0:
        _bt_name = result_selector.value.index[0]
        _navs, _comp, _turn, _meta = results[_bt_name]

        _held = _comp.columns[_comp.abs().sum() > 1e-6]
        _comp_held = _comp[_held]

        for _col in _held:
            comp_fig.add_trace(
                go.Scatter(
                    x=_comp_held.index,
                    y=_comp_held[_col],
                    mode="lines",
                    stackgroup="one",
                    name=_col,
                    hoveron="points+fills",
                )
            )

        comp_fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        comp_fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        comp_fig.update_layout(
            title=f"Stackplot: {_bt_name}",
            xaxis_title="Date",
            yaxis_title="Weight",
            template="plotly_white",
            yaxis=dict(range=[0, 1]),
        )

    comp_fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
