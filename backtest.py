"""Core backtesting engine: strategies, simulation, and performance metrics.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# Strategies (return target position per day) 

def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float).where(slow_ma.notna(), 0.0)


def rsi(prices: pd.Series, n: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_mean_reversion(prices: pd.Series, n: int = 14, low: int = 30, high: int = 70) -> pd.Series:
    """Buy when RSI < low, exit when RSI > high, otherwise hold the prior state."""
    r = rsi(prices, n)
    pos = pd.Series(np.nan, index=prices.index)
    pos[r < low] = 1.0
    pos[r > high] = 0.0
    return pos.ffill().fillna(0.0)


def momentum(prices: pd.Series, lookback: int = 126) -> pd.Series:
    """Long when trailing return over `lookback` days is positive."""
    return (prices.pct_change(lookback) > 0).astype(float)


STRATEGIES = {
    "SMA Crossover": (sma_crossover, {"fast": (5, 100, 20), "slow": (20, 300, 50)}),
    "RSI Mean Reversion": (rsi_mean_reversion, {"n": (5, 30, 14), "low": (10, 40, 30), "high": (60, 90, 70)}),
    "Momentum": (momentum, {"lookback": (20, 252, 126)}),
}


# Simulation 

def run_backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 5.0,
                 initial_capital: float = 10_000.0) -> pd.DataFrame:
    """Simulate holding `signal` (shifted one bar) with proportional trading costs."""
    returns = prices.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * returns - turnover * cost_bps / 1e4
    return pd.DataFrame({
        "price": prices,
        "signal": signal,
        "position": position,
        "strategy_ret": strat_ret,
        "benchmark_ret": returns,
        "strategy_equity": initial_capital * (1 + strat_ret).cumprod(),
        "benchmark_equity": initial_capital * (1 + returns).cumprod(),
    })


# Metrics 

def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def performance_metrics(ret: pd.Series, rf: float = 0.0) -> dict:
    ret = ret.dropna()
    equity = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    excess = ret - rf / TRADING_DAYS
    sharpe = excess.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (excess.mean() * TRADING_DAYS) / downside if downside and downside > 0 else np.nan
    max_dd = drawdown(equity).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Total Return": equity.iloc[-1] - 1,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }


def trade_stats(result: pd.DataFrame) -> dict:
    pos = result["position"]
    entries = ((pos > 0) & (pos.shift(1).fillna(0) == 0)).sum()
    return {"Trades": int(entries), "Time in Market": float((pos > 0).mean())}


# Extra analytics for the dashboard

VALID = {  # parameter combos that make no sense are skipped in sweeps
    "SMA Crossover": lambda p: p["fast"] < p["slow"],
    "RSI Mean Reversion": lambda p: p["low"] < p["high"],
}


def extract_trades(result: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip (compounded return while invested)."""
    held = result["position"] > 0
    block = (held != held.shift()).cumsum()
    rows = [{"Entry": seg.index[0], "Exit": seg.index[-1], "Days": len(seg),
             "Return": (1 + seg["strategy_ret"]).prod() - 1}
            for _, seg in result[held].groupby(block[held])]
    return pd.DataFrame(rows)


def rolling_sharpe(ret: pd.Series, window: int = 126) -> pd.Series:
    return ret.rolling(window).mean() / ret.rolling(window).std() * np.sqrt(TRADING_DAYS)


def monthly_returns(ret: pd.Series) -> pd.DataFrame:
    """Years as rows, months 1-12 as columns."""
    m = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    return m.unstack().reindex(columns=range(1, 13))


def sweep_sharpe(prices, func, base_params, grid, cost_bps=5.0, valid=None):
    """Sharpe for every combo in `grid` (1 or 2 params); other params stay at base_params.
    With 2 params: rows = first grid key, columns = second grid key."""
    names = list(grid)

    def sharpe(**kw):
        p = {**base_params, **kw}
        if valid and not valid(p):
            return np.nan
        r = run_backtest(prices, func(prices, **p), cost_bps)["strategy_ret"]
        return performance_metrics(r)["Sharpe"]

    if len(names) == 1:
        return pd.Series({v: sharpe(**{names[0]: v}) for v in grid[names[0]]}, name="Sharpe")
    a, b = names
    return pd.DataFrame({vb: {va: sharpe(**{a: va, b: vb}) for va in grid[a]} for vb in grid[b]})
