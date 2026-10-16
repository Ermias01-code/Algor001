import datetime as dt
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from backtest import (STRATEGIES, VALID, drawdown, extract_trades, monthly_returns, performance_metrics,
                      rolling_sharpe, rsi, run_backtest, sweep_sharpe, trade_stats)

st.set_page_config(page_title="Algo Trading Backtester", layout="wide")
st.title("Algorithmic Trading Backtester")


@st.cache_data(show_spinner="Downloading prices from Yahoo Finance...", ttl=3600)
def load_prices(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """Split/dividend-adjusted daily closes. Raises LookupError (bad ticker) or ConnectionError."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception:  # network error or rate limit: wait and retry
            time.sleep(1.5 * (attempt + 1))
            continue
        if df.empty:
            raise LookupError(ticker)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # newer yfinance returns a MultiIndex
            close = close.iloc[:, 0]
        return close.dropna().rename(ticker)
    raise ConnectionError("Yahoo Finance did not respond")


SEARCH_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}


@st.cache_data(show_spinner=False, ttl=3600)
def search_companies(query: str) -> dict:
    """Yahoo Finance search -> {display label: ticker}. Raises ConnectionError if the search fails."""
    try:
        quotes = yf.Search(query, max_results=10, news_count=0, lists_count=0).quotes
    except Exception as err:
        raise ConnectionError("Yahoo Finance search failed") from err
    matches = {}
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol or q.get("quoteType") not in SEARCH_TYPES:
            continue
        name = q.get("longname") or q.get("shortname") or symbol
        exchange = q.get("exchDisp") or q.get("exchange") or ""
        matches[f"{name} ({symbol}) · {exchange}".rstrip(" ·")] = symbol
    return matches


# Sidebar inputs 
with st.sidebar:
    st.header("Setup")
    query = st.text_input("Company", placeholder="Search by name or ticker, e.g. Tesla",
                          help="Type a company name or ticker, then press Enter.").strip()
    ticker = ""
    if query:
        try:
            matches = search_companies(query)
        except ConnectionError:
            matches = None
            ticker = query.upper()
            st.warning(f"Company search is unavailable right now, so I'll try '{ticker}' as a ticker symbol.")
        if matches:
            ticker = matches[st.selectbox("Select a match", list(matches))]
        elif matches is not None:
            st.warning(f"No matches for '{query}'. Try the company's full name or its ticker.")
    start = st.date_input("Start", dt.date(2015, 1, 1))
    end = st.date_input("End", dt.date.today())
    capital = st.number_input("Initial capital ($)", 1_000, 10_000_000, 10_000, step=1_000)
    cost_bps = st.slider("Trading cost (bps per unit turnover)", 0.0, 50.0, 5.0, 0.5)
    rf = st.number_input("Risk-free rate (annual, %)", 0.0, 15.0, 0.0, step=0.25) / 100

    st.header("Strategy")
    name = st.selectbox("Strategy", list(STRATEGIES))
    func, param_spec = STRATEGIES[name]
    params = {k: st.slider(k, lo, hi, default) for k, (lo, hi, default) in param_spec.items()}

# Run 
if not ticker:
    st.info("👈 Choose a company: search by name or ticker in the sidebar to run a backtest.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()
try:
    prices = load_prices(ticker, start, end)
except LookupError:
    st.error(f"No price data found for '{ticker}'. Check the ticker symbol and the date range.")
    st.stop()
except ConnectionError:
    st.error("Couldn't reach Yahoo Finance (it may be rate-limiting requests). Wait a minute and try again.")
    st.stop()
if len(prices) < 60:
    st.warning(f"Only {len(prices)} trading days of data for {ticker}. Widen the date range for a meaningful backtest.")
    st.stop()
st.caption(f"Data: {ticker} · {len(prices):,} trading days · {prices.index[0]:%b %d, %Y} to "
           f"{prices.index[-1]:%b %d, %Y} · Source: Yahoo Finance (adjusted for splits and dividends)")
if name == "SMA Crossover" and params["fast"] >= params["slow"]:
    st.warning("Fast window should be shorter than slow window.")
    st.stop()

result = run_backtest(prices, func(prices, **params), cost_bps=cost_bps, initial_capital=capital)
strat = performance_metrics(result["strategy_ret"], rf)
bench = performance_metrics(result["benchmark_ret"], rf)
trades = trade_stats(result)

# Headline metrics 
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return", f"{strat['Total Return']:.1%}", f"{strat['Total Return'] - bench['Total Return']:+.1%} vs B&H")
c2.metric("CAGR", f"{strat['CAGR']:.1%}")
c3.metric("Sharpe", f"{strat['Sharpe']:.2f}")
c4.metric("Max Drawdown", f"{strat['Max Drawdown']:.1%}")
c5.metric("Trades", trades["Trades"], f"{trades['Time in Market']:.0%} in market", delta_color="off")

# Dashboard
@st.cache_data(show_spinner="Running parameter sweep...")
def cached_sweep(prices, name, params, grid, cost_bps):
    return sweep_sharpe(prices, STRATEGIES[name][0], params, grid, cost_bps, VALID.get(name))


tab_over, tab_sig, tab_risk, tab_trades, tab_robust = st.tabs(
    ["Overview", "Signals", "Risk & Returns", "Trades", "Robustness"])

with tab_over:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=result.index, y=result["strategy_equity"], name="Strategy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["benchmark_equity"], name="Buy & Hold",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=drawdown(result["strategy_equity"]), name="Strategy DD",
                             fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=drawdown(result["benchmark_equity"]), name="B&H DD",
                             line=dict(dash="dot")), row=2, col=1)
    fig.update_yaxes(title_text="Portfolio value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Metrics")
        table = pd.DataFrame({"Strategy": strat, "Buy & Hold": bench})
        pct_rows = ["Total Return", "CAGR", "Volatility", "Max Drawdown"]
        fmt = table.copy().astype(object)
        for row in table.index:
            fmt.loc[row] = table.loc[row].map(lambda v: f"{v:.2%}" if row in pct_rows else f"{v:.2f}")
        st.dataframe(fmt)
    with right:
        st.subheader("Daily results")
        st.dataframe(result.tail(250).round(4))
        st.download_button("Download full results (CSV)", result.to_csv().encode(), f"{ticker}_{name}_backtest.csv")

with tab_sig:
    st.caption("Why the strategy trades when it does: colored line = invested, gray = in cash. "
               "Green/red triangles are the buy/sell signals (executed at that day's close).")
    has_panel = name != "SMA Crossover"
    fig = make_subplots(rows=2 if has_panel else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3] if has_panel else [1.0], vertical_spacing=0.05)
    invested = (result["signal"] > 0) | (result["position"] > 0)
    fig.add_trace(go.Scatter(x=result.index, y=result["price"], name="Price (cash)",
                             line=dict(color="lightgray", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["price"].where(invested), name="Price (invested)",
                             line=dict(width=2)), row=1, col=1)
    chg = result["signal"].diff().fillna(result["signal"])
    for mask, label, symbol, color in ((chg > 0, "Buy", "triangle-up", "green"), (chg < 0, "Sell", "triangle-down", "red")):
        fig.add_trace(go.Scatter(x=result.index[mask], y=result["price"][mask], mode="markers", name=label,
                                 marker=dict(symbol=symbol, size=10, color=color)), row=1, col=1)
    if name == "SMA Crossover":
        for w, label in ((params["fast"], "Fast SMA"), (params["slow"], "Slow SMA")):
            fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(w).mean(), name=f"{label} ({w})",
                                     line=dict(width=1.5, dash="dash")), row=1, col=1)
    elif name == "RSI Mean Reversion":
        fig.add_trace(go.Scatter(x=prices.index, y=rsi(prices, params["n"]), name="RSI"), row=2, col=1)
        for lvl in (params["low"], params["high"]):
            fig.add_hline(y=lvl, line_dash="dot", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=prices.index, y=prices.pct_change(params["lookback"]), name="Trailing return"),
                      row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", row=2, col=1)
        fig.update_yaxes(title_text="Trailing return", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig)

with tab_risk:
    window = st.slider("Rolling window (days)", 30, 252, 126)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rolling Sharpe ratio")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.index, y=rolling_sharpe(result["strategy_ret"], window), name="Strategy"))
        fig.add_trace(go.Scatter(x=result.index, y=rolling_sharpe(result["benchmark_ret"], window), name="Buy & Hold",
                                 line=dict(dash="dot")))
        fig.add_hline(y=0, line_color="gray")
        fig.update_layout(height=360, margin=dict(t=20), hovermode="x unified")
        st.plotly_chart(fig)
        st.caption("Shows when the strategy worked. A good one stays above zero, not just on average.")
    with c2:
        st.subheader("Distribution of daily returns")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=result["benchmark_ret"], name="Buy & Hold (all days)", opacity=0.6, nbinsx=80))
        fig.add_trace(go.Histogram(x=result.loc[result["position"] > 0, "strategy_ret"],
                                   name="Strategy (days invested)", opacity=0.6, nbinsx=80))
        fig.update_layout(barmode="overlay", height=360, margin=dict(t=20), xaxis_tickformat=".1%")
        st.plotly_chart(fig)
        st.caption("Compares the shape of returns: does the strategy avoid the fat left tail?")
    st.subheader("Monthly returns (strategy)")
    mr = monthly_returns(result["strategy_ret"])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(z=mr.values, x=months, y=[str(y) for y in mr.index], colorscale="RdYlGn", zmid=0,
                               texttemplate="%{z:.1%}", colorbar_title="Return"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(300, 40 * len(mr) + 100), margin=dict(t=20))
    st.plotly_chart(fig)

with tab_trades:
    tr = extract_trades(result)
    if tr.empty:
        st.info("The strategy made no trades in this period.")
    else:
        a, b, c = st.columns(3)
        a.metric("Win rate", f"{(tr['Return'] > 0).mean():.0%}")
        b.metric("Average trade", f"{tr['Return'].mean():.2%}")
        c.metric("Average holding (days)", f"{tr['Days'].mean():.0f}")
        fig = go.Figure(go.Bar(x=tr["Entry"], y=tr["Return"],
                               marker_color=np.where(tr["Return"] > 0, "seagreen", "indianred").tolist()))
        fig.update_layout(height=360, margin=dict(t=20), yaxis_tickformat=".0%", yaxis_title="Return per trade")
        st.plotly_chart(fig)
        st.caption("Many small wins plus a few large losses (or the reverse) is the 'personality' of a strategy.")
        st.dataframe(tr)

with tab_robust:
    st.subheader("1. Parameter sensitivity (Sharpe ratio)")
    names = list(param_spec)
    if len(names) >= 2:
        c1, c2 = st.columns(2)
        x_param = c1.selectbox("X-axis parameter", names, index=0)
        y_param = c2.selectbox("Y-axis parameter", [n for n in names if n != x_param], index=0)
        grid = {k: sorted(set(np.linspace(param_spec[k][0], param_spec[k][1], 10).astype(int).tolist()))
                for k in (y_param, x_param)}
        z = cached_sweep(prices, name, params, grid, cost_bps)
        fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                                   colorscale="RdYlGn", zmid=0, colorbar_title="Sharpe"))
        fig.update_layout(xaxis_title=x_param, yaxis_title=y_param, height=450, margin=dict(t=20))
    else:
        k = names[0]
        vals = sorted(set(np.linspace(param_spec[k][0], param_spec[k][1], 15).astype(int).tolist()))
        z = cached_sweep(prices, name, params, {k: vals}, cost_bps)
        fig = go.Figure(go.Scatter(x=z.index, y=z.values, mode="lines+markers"))
        fig.update_layout(xaxis_title=k, yaxis_title="Sharpe", height=400, margin=dict(t=20))
    st.plotly_chart(fig)
    st.caption("Look for a broad green plateau. A single bright spot surrounded by red usually means "
               "the parameters are overfit to the past.")

    st.subheader("2. Effect of trading costs")
    sig = func(prices, **params)
    fig = go.Figure()
    for bps in (0, 5, 10, 25, 50):
        r = run_backtest(prices, sig, bps, capital)
        fig.add_trace(go.Scatter(x=r.index, y=r["strategy_equity"], name=f"{bps} bps"))
    fig.add_trace(go.Scatter(x=result.index, y=result["benchmark_equity"], name="Buy & Hold",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(height=400, margin=dict(t=20), hovermode="x unified", yaxis_title="Portfolio value ($)")
    st.plotly_chart(fig)
    st.caption("Same signals, different costs. Frequent traders lose the most as costs rise.")

    st.subheader("3. All strategies side by side (default parameters)")
    fig = go.Figure()
    rows = {}
    for s_name, (s_func, s_spec) in STRATEGIES.items():
        r = run_backtest(prices, s_func(prices, **{k: v[2] for k, v in s_spec.items()}), cost_bps, capital)
        fig.add_trace(go.Scatter(x=r.index, y=r["strategy_equity"], name=s_name))
        m = performance_metrics(r["strategy_ret"], rf)
        rows[s_name] = {"CAGR": f"{m['CAGR']:.1%}", "Sharpe": f"{m['Sharpe']:.2f}",
                        "Max Drawdown": f"{m['Max Drawdown']:.1%}"}
    fig.add_trace(go.Scatter(x=result.index, y=result["benchmark_equity"], name="Buy & Hold",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(height=400, margin=dict(t=20), hovermode="x unified", yaxis_title="Portfolio value ($)")
    st.plotly_chart(fig)
    st.dataframe(pd.DataFrame(rows).T)

st.caption("Signals use data through day t and are traded from day t+1 (no look-ahead). "
           "Historical results do not predict future performance.")
