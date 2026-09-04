from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import backtest, datafeed, exodata, features, forwardtest, journal, models, optimize
from src.config import load_settings
from src.features import FEATURE_GROUPS, build_live_row, build_panel
from src.models import Genome

st.set_page_config(page_title="Adaptive ML Trader", page_icon="🧠", layout="wide")
settings = load_settings()
DATA_DIR = settings.data_dir
JOURNAL_PATH = os.path.join(DATA_DIR, "journal.db")
SNAP_PATH = os.path.join(DATA_DIR, "snapshots.csv")
CHAMPION_PATH = os.path.join(settings.model_dir, "champion.joblib")
CHAMPION_JSON = os.path.join(settings.model_dir, "champion_genome.json")


# ---------------------------------------------------------------- helpers
def sig_label(v: float) -> str:
    if v >= 0.33:
        return "🟢 BUY"
    if v <= -0.33:
        return "🔴 SELL"
    return "⚪ HOLD"


def gauge(title: str, v: float) -> go.Figure:
    color = "#00C853" if v > 0.15 else "#FF5252" if v < -0.15 else "#9E9E9E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(v, 3), number={"font": {"size": 28}},
        gauge={"axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
               "bar": {"color": color, "thickness": 0.25}, "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [-1, -0.33], "color": "rgba(255,82,82,.25)"},
                         {"range": [-0.33, 0.33], "color": "rgba(158,158,158,.15)"},
                         {"range": [0.33, 1], "color": "rgba(0,200,83,.25)"}]},
        title={"text": title, "font": {"size": 15}}))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=45, b=5),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbols, asset_pairs, start, end, backend):
    am = dict(asset_pairs)
    bars = datafeed.get_daily_bars(list(symbols), am, start, end, settings)
    macro = datafeed.get_macro(start, settings)
    sent = datafeed.get_news_sentiment(list(symbols), settings, days=45, backend=backend)
    exo = exodata.get_exo(start, SNAP_PATH)
    return bars, macro, sent, exo


@st.cache_data(ttl=21600, show_spinner=False)
def cached_wf(panel, genome_dict):
    return models.walk_forward_signals(panel, Genome(**genome_dict))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧠 Adaptive ML Trader")
    st.markdown(f"**Alpaca:** {'🟢 connected' if datafeed.has_alpaca(settings) else '🔴 no keys — yfinance fallback'}")
    st.markdown(f"**Mode:** {'📄 PAPER' if settings.paper else '⚠️ LIVE TRADING'}")
    st.markdown(f"**Macro:** {'FRED' if settings.fred_key else 'Yahoo proxies'} · **Sentiment:** `{settings.sentiment_backend}`")
    st.divider()
    stocks = [s.strip().upper() for s in st.text_input(
        "Stocks (comma-separated)", "AAPL, MSFT, NVDA, SPY").split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in st.text_input(
        "Crypto (comma-separated)", "BTC/USD, ETH/USD").split(",") if s.strip()]
    symbols = stocks + cryptos
    asset_map = {**{s: "stock" for s in stocks}, **{s: "crypto" for s in cryptos}}
    years = st.slider("History (years)", 1, 8, 4)
    start = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=365 * years)
    end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    with st.expander("Costs & risk"):
        fee_bps = st.number_input("Fees (bps)", 0.0, 50.0, 2.0, 0.5)
        slip_bps = st.number_input("Slippage (bps)", 0.0, 50.0, 5.0, 0.5)
        allow_short = st.checkbox("Allow shorting (stocks only)", value=False)

if not symbols:
    st.error("Add at least one symbol in the sidebar.")
    st.stop()

with st.spinner("Loading market data…"):
    bars, macro, sent, exo = load_data(tuple(symbols), tuple(sorted(asset_map.items())),
                                       start, end, settings.sentiment_backend)
bars = {s: df for s, df in bars.items() if len(df) > 120}
if not bars:
    st.error("No data returned. Check symbols and API keys.")
    st.stop()
closes = pd.DataFrame({s: df["close"] for s, df in bars.items()})

# champion: joblib bundle, else genome JSON (CI/Cloud) with model trained on the fly
champion = models.load_bundle(CHAMPION_PATH)
if champion is None and os.path.exists(CHAMPION_JSON):
    try:
        champion = {"genome": Genome(**json.load(open(CHAMPION_JSON))), "model": None}
    except Exception:
        champion = None

genome = champion["genome"] if champion else Genome()
panel = build_panel(bars, macro, sent, genome, exo)
with st.spinner("Training adaptive models (walk-forward — first run may take ~30s)…"):
    sig, last_model = cached_wf(panel, asdict(genome))
live_model = (champion.get("model") if champion else None) or last_model

tab_daily, tab_live, tab_lab, tab_bt, tab_paper, tab_auto = st.tabs(
    ["📊 Daily Signals", "⚡ Realtime", "🧬 Strategy Lab", "📈 Backtest",
     "💸 Paper Trading", "🤖 Automation"])

# ================================================================ DAILY
with tab_daily:
    st.caption(f"Active strategy: **{genome.name}** · horizon {genome.horizon}d · "
               f"retrain every {genome.retrain_every}d · dead-zone ±{genome.entry:.2f} · "
               f"macro {'✅' if genome.use_macro else '❌'} · sentiment {'✅' if genome.use_sentiment else '❌'}"
               + ("" if champion else " · *(default genome — promote one in the Lab)*"))

    if not macro.empty:
        m = macro.iloc[-1]
        c = st.columns(6)
        c[0].metric("VIX", f"{m.get('vix_lvl', np.nan):.1f}", f"{m.get('vix_chg5', 0):+.1%}")
        c[1].metric("VIX z-score", f"{m.get('vix_z', np.nan):+.2f}")
        c[2].metric("10Y Δ5d", f"{m.get('y10_chg5', np.nan):+.2f}")
        c[3].metric("SPY 21d", f"{m.get('spy_ret21', np.nan):+.1%}")
        c[4].metric("Oil 5d", f"{m.get('oil_ret5', np.nan):+.1%}")
        vz, sr = m.get("vix_z", 0), m.get("spy_ret21", 0)
        regime = "🟢 Risk-on" if (vz < 0 and sr > 0) else ("🔴 Risk-off" if (vz > 1 or sr < -0.03) else "⚪ Neutral")
        c[5].markdown(f"### {regime}")

    if sig.empty:
        st.warning("Not enough history to train — increase the history slider.")
        st.stop()

    st.markdown("### Latest signals")
    st.caption("The signal **is** the position size: +1 = max long, −1 = max short, 0 = flat. "
               "Everything between is a linear blend — e.g. +0.4 = 40% of max long allocation.")
    latest = sig.iloc[-1].dropna()
    cols = st.columns(min(4, len(latest)))
    for i, (sym, v) in enumerate(latest.items()):
        with cols[i % len(cols)]:
            st.plotly_chart(gauge(sym, float(v)), use_container_width=True)
            st.markdown(f"<p style='text-align:center'>{sig_label(v)} · strength {abs(v):.0%}</p>",
                        unsafe_allow_html=True)

    w = latest * (genome.max_leverage / len(latest))
    st.dataframe(pd.DataFrame({"signal": latest, "target weight": w,
                               "action": latest.map(sig_label)})
                 .style.format({"signal": "{:+.2f}", "target weight": "{:+.1%}"}),
                 use_container_width=True)

    st.markdown("### Signal history (out-of-sample)")
    fig = px.imshow(sig.tail(90).T, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                    aspect="auto", labels=dict(color="signal"))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if last_model is not None and not last_model.feature_importance().empty:
        st.markdown("### What drives the model")
        fi = last_model.feature_importance().tail(20)
        st.plotly_chart(px.bar(fi, orientation="h", height=420), use_container_width=True)

# ================================================================ REALTIME
with tab_live:
    st.subheader("⚡ Realtime signals")
    st.caption("Today's intraday action is folded into a synthetic daily bar and scored by the active model.")
    if live_model is None:
        st.warning("No trained model yet — open the Daily tab once, or promote a champion in the Lab.")
    else:
        if datafeed.has_alpaca(settings):
            try:
                clock = datafeed.trading_client(settings).get_clock()
                st.info(f"US stock market {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'} — crypto trades 24/7")
            except Exception:
                pass
        refresh = st.select_slider("Auto-refresh (seconds)", [15, 30, 60, 120, 300], 60)

        def render_live():
            for sym in bars:
                try:
                    price = datafeed.get_latest_price(sym, asset_map[sym], settings)
                    intr = datafeed.get_intraday_bars(sym, asset_map[sym], settings)
                    row = build_live_row(sym, bars[sym], intr, macro, sent, genome, exo)
                    v = float(live_model.predict_signal(row)[0])
                    prev = float(bars[sym]["close"].iloc[-1])
                    chg = price / prev - 1 if prev and price == price else np.nan
                except Exception:
                    price, chg, v = np.nan, np.nan, np.nan
                c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
                c1.markdown(f"#### {sym}")
                c2.metric("Price", f"{price:,.2f}" if price == price else "—",
                          f"{chg:+.2%} vs prev close" if chg == chg else None)
                c3.markdown(f"### {sig_label(v)}" if v == v else "—")
                with c4:
                    if v == v:
                        st.plotly_chart(gauge("", v), use_container_width=True, key=f"live_{sym}")
            st.caption(f"Updated {dt.datetime.now(dt.timezone.utc).replace(tzinfo=None):%Y-%m-%d %H:%M:%S} UTC")

        if hasattr(st, "fragment"):
            st.fragment(run_every=dt.timedelta(seconds=refresh))(render_live)()
        else:
            render_live()
            if st.button("🔄 Refresh"):
                st.rerun()

# ================================================================ STRATEGY LAB
with tab_lab:
    st.subheader("🧬 Strategy Lab — create strategies that adapt to the market")
    st.caption("Each candidate = model type + feature groups (incl. options/short/on-chain) + "
               "macro/sentiment toggles + horizon + thresholds + sizing. "
               "Fitness = Sharpe − ½·|MaxDD| on a strictly out-of-sample walk-forward backtest.")
    method = st.radio("Search method", ["Evolution (genetic)", "Optuna (Bayesian)"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    eval_years = c1.slider("Evaluation window (years)", 1, 4, 2)
    seed = c2.number_input("Random seed", 0, 9999, 42)
    if method.startswith("Evolution"):
        pop_size = c3.slider("Population", 6, 24, 10)
        n_gen = st.slider("Generations", 1, 12, 4)
    else:
        n_trials = c3.slider("Trials", 10, 200, 40)
        c4, c5 = st.columns(2)
        timeout_min = c4.slider("Timeout (minutes, 0 = none)", 0, 120, 30)
        resume = c5.checkbox("Resume saved study (data/optuna.db)", True)

    if st.button("🚀 Run search", type="primary"):
        full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
        cutoff = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=365 * eval_years)
        panel_full = build_panel(bars, macro, sent, full, exo)
        panel_full = panel_full[panel_full["date"] >= pd.Timestamp(cutoff)]
        bar, status = st.progress(0.0), st.empty()
        if method.startswith("Evolution"):
            def cb(g, n, best_score, best_g):
                bar.progress((g + 1) / n)
                status.write(f"Generation {g + 1}/{n} — best fitness **{best_score:.3f}** ({best_g.name})")
            rows, hist = models.evolve(panel_full, pop_size, n_gen, seed,
                                       cost_bps=fee_bps + slip_bps, progress=cb)
        else:
            base = {}
            def cb2(study, trial):
                base.setdefault("n", len(study.trials) - 1)   # trials already in the study before this click
                done = len(study.trials) - base["n"]
                bar.progress(min(1.0, done / max(1, n_trials)))
                carried = f" · {base['n']} carried over from a previous resumed run" if base["n"] > 0 else ""
                try:
                    status.write(f"Trial {done}/{n_trials} — best {study.best_value:.3f}{carried}")
                except Exception:
                    status.write(f"Trial {done}/{n_trials}{carried}")
            storage = f"sqlite:///{os.path.join(DATA_DIR, 'optuna.db')}" if resume else None
            rows, hist, _ = optimize.optimize(panel_full, n_trials=n_trials, seed=seed,
                                              cost_bps=fee_bps + slip_bps, storage=storage,
                                              timeout=timeout_min * 60 or None, progress=cb2)
        st.session_state["lab"] = {"rows": rows, "hist": hist}
        status.success("Done — pick a champion below.")

    if "lab" in st.session_state and st.session_state["lab"]["rows"]:
        rows = st.session_state["lab"]["rows"]
        hist = st.session_state["lab"]["hist"]
        st.plotly_chart(px.line(hist, x="generation", y=["best", "avg"], markers=True,
                                title="Fitness over search"), use_container_width=True)
        st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "genome"}
                                   for r in rows]), use_container_width=True)
        idx = st.number_input("Leaderboard row to promote", 0, len(rows) - 1, 0)
        if st.button("💾 Train on full history & save as champion"):
            g = rows[int(idx)]["genome"]
            full = Genome(feature_groups=tuple(FEATURE_GROUPS), use_macro=True, use_sentiment=True)
            with st.spinner("Training final model on all history…"):
                m = models.train_final_model(build_panel(bars, macro, sent, full, exo), g)
            models.save_bundle(CHAMPION_PATH, g, m)
            os.makedirs(settings.model_dir, exist_ok=True)
            json.dump(asdict(g), open(CHAMPION_JSON, "w"), indent=2)
            st.success(f"Saved **{g.name}** → `{CHAMPION_PATH}`. Other tabs use it after a rerun.")

# ================================================================ BACKTEST
with tab_bt:
    st.subheader("Backtest (walk-forward, out-of-sample)")
    src = st.radio("Strategy source", ["Champion", "Custom", "Default"], horizontal=True)
    if src == "Champion" and champion:
        g_bt = champion["genome"]
    elif src == "Custom":
        c1, c2, c3 = st.columns(3)
        groups = c1.multiselect("Feature groups", list(FEATURE_GROUPS),
                                default=["trend", "momentum", "volatility"])
        use_m = c2.checkbox("Macro features", True)
        use_s = c3.checkbox("Sentiment features", True)
        c4, c5, c6, c7 = st.columns(4)
        g_bt = Genome(name="custom", feature_groups=tuple(groups) or ("momentum",),
                      use_macro=use_m, use_sentiment=use_s,
                      horizon=c4.selectbox("Horizon", [3, 5, 10], 1),
                      entry=c5.slider("Dead zone", 0.0, 0.5, 0.15),
                      smooth=c6.slider("Smoothing", 1, 10, 3),
                      max_leverage=c7.slider("Max leverage", 0.25, 2.0, 1.0, 0.25))
    else:
        g_bt = Genome()

    if st.button("▶️ Run backtest", type="primary"):
        with st.spinner("Running walk-forward backtest…"):
            p = build_panel(bars, macro, sent, g_bt, exo)
            s_bt, _ = cached_wf(p, asdict(g_bt))
            res = backtest.run_backtest(closes, s_bt, fee_bps, slip_bps,
                                        g_bt.max_leverage, allow_short)
        st.session_state["bt"] = {"res": res, "sig": s_bt, "name": g_bt.name}

    if "bt" in st.session_state:
        res, s_bt = st.session_state["bt"]["res"], st.session_state["bt"]["sig"]
        m = res.metrics
        cols = st.columns(7)
        for col, (k, fmt) in zip(cols, [("total_return", "{:.1%}"), ("cagr", "{:.1%}"),
                                        ("sharpe", "{:.2f}"), ("sortino", "{:.2f}"),
                                        ("max_drawdown", "{:.1%}"), ("hit_rate", "{:.1%}"),
                                        ("exposure", "{:.1%}")]):
            col.metric(k.replace("_", " ").title(), fmt.format(m[k]))

        bench = (1 + closes.pct_change().mean(axis=1).fillna(0)).cumprod().reindex(res.equity.index).ffill()
        fig = go.Figure()
        fig.add_scatter(x=res.equity.index, y=res.equity, name="Strategy", line=dict(color="#00C853"))
        fig.add_scatter(x=bench.index, y=bench, name="Equal-weight buy & hold",
                        line=dict(color="#9E9E9E", dash="dash"))
        fig.update_layout(title="Equity curve", height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.area(res.drawdown, title="Drawdown")
        fig2.update_layout(height=200, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)

        sym = st.selectbox("Inspect symbol", list(s_bt.columns))
        pr, s1 = closes[sym].reindex(s_bt.index), s_bt[sym]
        buy = (s1 > g_bt.entry) & (s1.shift(1) <= g_bt.entry)
        sell = (s1 < -g_bt.entry) & (s1.shift(1) >= -g_bt.entry)
        fig3 = go.Figure()
        fig3.add_scatter(x=pr.index, y=pr, name=sym, line=dict(color="#4C9AFF"))
        fig3.add_scatter(x=pr.index[buy.fillna(False)], y=pr[buy.fillna(False)], mode="markers",
                         name="Enter long", marker=dict(color="#00C853", size=9, symbol="triangle-up"))
        fig3.add_scatter(x=pr.index[sell.fillna(False)], y=pr[sell.fillna(False)], mode="markers",
                         name="Enter short", marker=dict(color="#FF5252", size=9, symbol="triangle-down"))
        fig3.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig3, use_container_width=True)

# ================================================================ PAPER TRADING
with tab_paper:
    st.subheader("Forward test — Alpaca paper account")
    if not datafeed.has_alpaca(settings):
        st.warning("Add `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` to enable paper trading.")
        st.stop()
    tc = datafeed.trading_client(settings)
    try:
        acct = tc.get_account()
    except Exception as e:
        st.error(f"Could not reach Alpaca: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${float(acct.equity):,.2f}")
    c2.metric("Cash", f"${float(acct.cash):,.2f}")
    c3.metric("Buying power", f"${float(acct.buying_power):,.2f}")

    pos = tc.get_all_positions()
    if pos:
        st.dataframe(pd.DataFrame([{
            "symbol": p.symbol, "qty": float(p.qty), "value": float(p.market_value),
            "avg entry": float(p.avg_entry_price), "price": float(p.current_price),
            "unrealized P&L": float(p.unrealized_pl)} for p in pos]),
            use_container_width=True)
    else:
        st.info("No open positions.")

    st.markdown("### Rebalance to current signals")
    if sig.empty:
        st.warning("No signals available — check the Daily tab first.")
        st.stop()
    latest_p = sig.iloc[-1].reindex(list(bars.keys())).fillna(0)
    w = latest_p.clip(lower=0 if not allow_short else -1)
    crypto_syms = [s for s in w.index if asset_map.get(s) == "crypto"]
    w[crypto_syms] = w[crypto_syms].clip(lower=0)      # no crypto shorts on Alpaca
    w = w * (genome.max_leverage / len(w))
    targets = (w * float(acct.equity)).to_dict()
    cur = {p.symbol: float(p.market_value) for p in pos}
    preview = pd.DataFrame({
        "signal": latest_p, "target $": pd.Series(targets),
        "current $": [cur.get(s, cur.get(s.replace("/", ""), 0.0)) for s in targets]})
    preview["trade $"] = preview["target $"] - preview["current $"]
    st.dataframe(preview.style.format("{:+.2f}"), use_container_width=True)

    confirm = st.checkbox("I understand this submits orders to my **paper** account")
    if st.button("📤 Execute rebalance", disabled=not confirm, type="primary"):
        results = datafeed.rebalance(tc, targets, asset_map)
        journal.log(JOURNAL_PATH, "signals", latest_p.to_dict())
        journal.log(JOURNAL_PATH, "orders", results)
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.success("Orders submitted and journaled.")

    with st.expander("📒 Forward-test journal"):
        st.dataframe(journal.read(JOURNAL_PATH, limit=100), use_container_width=True)

# ================================================================ AUTOMATION
with tab_auto:
    st.subheader("🤖 Automation status & continuous forward test")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Latest pipeline output")
        p = os.path.join(DATA_DIR, "latest_signals.json")
        if os.path.exists(p):
            d = json.load(open(p))
            ts = pd.Timestamp(d.get("ts"))
            if ts.tzinfo is not None:          # be robust to either naive or tz-aware stored timestamps
                ts = ts.tz_localize(None)
            age = pd.Timestamp.now("UTC").tz_localize(None) - ts
            st.metric("Last pipeline run", f"{ts:%Y-%m-%d %H:%M} UTC",
                      f"{age.days * 24 + age.seconds // 3600}h ago")
            st.write(f"**Genome:** {d.get('genome', '?')} · **Regime:** `{d.get('regime')}`")
            st.json(d.get("signals", {}), expanded=False)
        else:
            st.info("No pipeline output yet — run it once (right) or start the scheduler.")
    with right:
        st.markdown("#### Run the daily pipeline now")
        st.caption("Trains the champion, logs signals, snapshots exo data, and rebalances "
                   "the paper account (per universe.json).")
        ack = st.checkbox("I understand this may submit paper orders", key="pipe_ack")
        if st.button("▶️ Run pipeline", disabled=not ack):
            if not os.path.exists("universe.json"):
                st.error("universe.json not found — copy universe.example.json and adjust it.")
            else:
                with st.spinner("Running pipeline (1–3 min)…"):
                    from src.pipeline import run as pipeline_run
                    try:
                        out = pipeline_run("universe.json")
                        st.success("Pipeline finished.")
                        st.json(out)
                    except Exception as e:
                        st.error(str(e))

    st.divider()
    st.markdown("#### Recent automation events")
    st.dataframe(journal.read(JOURNAL_PATH, limit=50), use_container_width=True)

    evo_events = journal.read(JOURNAL_PATH, kind="evolution", limit=200)
    if not evo_events.empty:
        st.markdown("#### 🧬 Autonomous lab history")
        st.caption("Every time the lab ran: what triggered it (scheduled / ic_panic / manual), "
                   "champion vs challenger score, the champion's live forward-test IC at "
                   "decision time, and whether a promotion happened.")
        ev = pd.DataFrame([json.loads(p) for p in evo_events["payload"]])
        ev.insert(0, "ts", pd.to_datetime(evo_events["ts"]).dt.strftime("%Y-%m-%d %H:%M"))
        st.dataframe(ev, use_container_width=True)

    st.divider()
    st.markdown("#### Continuous forward test — do journaled signals predict future returns?")
    events = forwardtest.signal_events(journal.read(JOURNAL_PATH, limit=10000))
    rep = forwardtest.report(events, closes, horizon=genome.horizon)
    if rep.get("n", 0) >= 10:
        c1, c2, c3 = st.columns(3)
        c1.metric("Matured observations", rep["n"])
        c2.metric("Information coefficient (Spearman)", f"{rep['ic']:.3f}")
        verdict = "✅ positive edge" if rep["ic"] > 0.02 else ("❌ negative" if rep["ic"] < -0.02 else "⏳ inconclusive")
        c3.metric("Verdict", verdict)
        st.dataframe(rep["by_bucket"], use_container_width=True)
        if not rep["rolling"].empty:
            st.plotly_chart(px.line(rep["rolling"], x="date", y="ic_roll",
                                    title="Rolling IC (rank correlation)"), use_container_width=True)
        st.plotly_chart(px.scatter(rep["detail"], x="signal", y="fwd_ret", color="symbol",
                                   trendline="ols", title="Signal vs realized forward return"),
                        use_container_width=True)
    else:
        st.info(f"Only {rep.get('n', 0)} matured observations so far — "
                "the scheduler/pipeline accumulates them daily.")
