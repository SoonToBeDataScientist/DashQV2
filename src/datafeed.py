from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd

from .config import Settings

_clients: dict = {}


def has_alpaca(s: Settings) -> bool:
    return bool(s.alpaca_key and s.alpaca_secret)


def _stock_client(s: Settings):
    if "stock" not in _clients:
        from alpaca.data.historical import StockHistoricalDataClient
        _clients["stock"] = StockHistoricalDataClient(s.alpaca_key, s.alpaca_secret)
    return _clients["stock"]


def _crypto_client(s: Settings):
    if "crypto" not in _clients:
        from alpaca.data.historical import CryptoHistoricalDataClient
        _clients["crypto"] = CryptoHistoricalDataClient(s.alpaca_key, s.alpaca_secret)
    return _clients["crypto"]


def _news_client(s: Settings):
    if "news" not in _clients:
        from alpaca.data.historical.news import NewsClient
        _clients["news"] = NewsClient(s.alpaca_key, s.alpaca_secret)
    return _clients["news"]


def trading_client(s: Settings):
    if "trading" not in _clients:
        from alpaca.trading.client import TradingClient
        _clients["trading"] = TradingClient(s.alpaca_key, s.alpaca_secret, paper=s.paper)
    return _clients["trading"]


# ---------------------------------------------------------------- bars
def _clean_bars(df: pd.DataFrame, daily: bool) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.normalize() if daily else idx
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _yf_symbol(sym: str) -> str:
    return sym.replace("/", "-")  # BTC/USD -> BTC-USD


def _yf_bars(sym: str, start, end, interval: str):
    try:
        import yfinance as yf
        df = yf.download(_yf_symbol(sym), start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(how="all")
    except Exception:
        return None


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _fetch_daily(symbols, asset_map, start, end, s: Settings) -> dict:
    """One batch download. Stocks come split+dividend ADJUSTED from Alpaca (the API default is
    raw, which turns a 10:1 split into a -90% day) and from the consolidated SIP feed, which the
    free plan serves for anything older than 15 min — IEX alone is ~2% of real volume. If the
    plan refuses SIP, fall back to IEX. yfinance (auto-adjusted) fills whatever is still missing.
    Each frame's .attrs["source"] records where it came from."""
    out = {}
    # Alpaca only covers US-listed stocks + majors crypto — anything with an exchange suffix
    # (e.g. ".JK" for IDX) has zero Alpaca coverage, so keep it out of that batch entirely.
    stocks = [x for x in symbols if asset_map.get(x) == "stock" and "." not in x]
    cryptos = [x for x in symbols if asset_map.get(x) == "crypto"]
    if has_alpaca(s):
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        if stocks:
            sip_end = min(end, _utcnow() - dt.timedelta(minutes=16))
            for feed, e in ((DataFeed.SIP, sip_end), (DataFeed.IEX, end)):
                try:
                    df = _stock_client(s).get_stock_bars(StockBarsRequest(
                        symbol_or_symbols=stocks, timeframe=TimeFrame.Day, start=start, end=e,
                        feed=feed, adjustment=Adjustment.ALL)).df
                except Exception:
                    continue
                for sym in stocks:
                    try:
                        out[sym] = _clean_bars(df.xs(sym, level="symbol"), daily=True)
                        out[sym].attrs["source"] = f"alpaca-{feed.value}"
                    except (KeyError, TypeError):
                        pass
                break
        if cryptos:
            try:
                df = _crypto_client(s).get_crypto_bars(CryptoBarsRequest(
                    symbol_or_symbols=cryptos, timeframe=TimeFrame.Day, start=start, end=end)).df
                for sym in cryptos:
                    try:
                        out[sym] = _clean_bars(df.xs(sym, level="symbol"), daily=True)
                        out[sym].attrs["source"] = "alpaca"
                    except (KeyError, TypeError):
                        pass
            except Exception:
                pass
    for sym in symbols:  # yfinance fallback for anything missing
        if sym not in out:
            df = _yf_bars(sym, start, end, "1d")
            if df is not None and not df.empty:
                out[sym] = _clean_bars(df, daily=True)
                out[sym].attrs["source"] = "yfinance"
    return out


def _cache_path(cache_dir, sym):
    return os.path.join(cache_dir, sym.replace("/", "_") + ".pkl")


def get_daily_bars(symbols, asset_map, start, end, s: Settings, full_refresh_days=7) -> dict:
    """Daily bars per symbol. With s.bars_cache_dir set (CI restores it via actions/cache), only
    the last few days are downloaded; the rest comes from the cache. A symbol is re-downloaded in
    full when its cache is older than full_refresh_days, doesn't reach back to `start`, or when an
    already-cached close changed — i.e. a split/dividend re-adjusted its history."""
    cache_dir = s.bars_cache_dir
    if not cache_dir:
        return _fetch_daily(symbols, asset_map, start, end, s)
    now, t0 = pd.Timestamp(_utcnow()), pd.Timestamp(start)
    cached = {}
    for sym in symbols:
        try:
            c = pd.read_pickle(_cache_path(cache_dir, sym))
            if (now - c["full_at"] < pd.Timedelta(days=full_refresh_days)
                    and c["bars"].index[0] <= t0 + pd.Timedelta(days=7)):
                cached[sym] = c
        except Exception:
            pass

    def save(sym, bars, full_at, source):
        os.makedirs(cache_dir, exist_ok=True)
        pd.to_pickle({"bars": bars, "full_at": full_at, "source": source}, _cache_path(cache_dir, sym))

    out, full = {}, [x for x in symbols if x not in cached]
    if cached:
        inc_start = min(c["bars"].index[-1] for c in cached.values()) - pd.Timedelta(days=10)
        fresh = _fetch_daily(list(cached), asset_map, inc_start.to_pydatetime(), end, s)
        for sym, c in cached.items():
            old, new = c["bars"], fresh.get(sym)
            if new is None or new.empty:
                out[sym] = old                       # download failed: stale data, caught downstream
                out[sym].attrs["source"] = c.get("source", "?") + "+cache-stale"
                continue
            ov = old.index.intersection(new.index)
            ov = ov[ov < old.index[-1]]              # the newest cached bar may have been partial
            if len(ov) and (new.loc[ov, "close"] / old.loc[ov, "close"] - 1).abs().max() > 0.005:
                full.append(sym)
                continue
            merged = pd.concat([old[old.index < new.index[0]], new])
            save(sym, merged, c["full_at"], new.attrs.get("source", "?"))
            out[sym] = merged
            out[sym].attrs["source"] = new.attrs.get("source", "?") + "+cache"
    if full:
        for sym, df in _fetch_daily(full, asset_map, start, end, s).items():
            save(sym, df, now, df.attrs.get("source", "?"))
            out[sym] = df
    for sym, df in out.items():
        src = df.attrs.get("source")
        out[sym] = df[df.index >= t0.normalize()]
        out[sym].attrs["source"] = src
    return out


def get_intraday_bars(sym, asset_class, s: Settings, minutes=15, days=3) -> pd.DataFrame:
    end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start = end - dt.timedelta(days=days)
    if has_alpaca(s):
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            tf = TimeFrame(minutes, TimeFrameUnit.Minute)
            if asset_class == "stock":
                req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=tf, start=start,
                                       end=end, feed=DataFeed.IEX)
                df = _stock_client(s).get_stock_bars(req).df
            else:
                req = CryptoBarsRequest(symbol_or_symbols=[sym], timeframe=tf, start=start, end=end)
                df = _crypto_client(s).get_crypto_bars(req).df
            if not df.empty:
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(sym, level="symbol")
                return _clean_bars(df, daily=False)
        except Exception:
            pass
    df = _yf_bars(sym, start, end, f"{minutes}m")
    return _clean_bars(df, daily=False) if df is not None and not df.empty else pd.DataFrame()


def get_latest_price(sym, asset_class, s: Settings) -> float:
    if has_alpaca(s):
        try:
            if asset_class == "stock":
                from alpaca.data.enums import DataFeed
                from alpaca.data.requests import StockLatestTradeRequest
                r = _stock_client(s).get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=[sym], feed=DataFeed.IEX))
            else:
                from alpaca.data.requests import CryptoLatestTradeRequest
                r = _crypto_client(s).get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=[sym]))
            return float(r[sym].price)
        except Exception:
            pass
    df = _yf_bars(sym, dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=7), None, "1d")
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else float("nan")


# ---------------------------------------------------------------- macro
FRED_SERIES = {"vix": "VIXCLS", "us10y": "DGS10", "yc_spread": "T10Y2Y",
               "unrate": "UNRATE", "consumer_sent": "UMCSENT", "oil": "DCOILWTICO"}
YF_MACRO = {"vix": "^VIX", "us10y": "^TNX", "dxy": "DX-Y.NYB",
            "spy": "SPY", "gold": "GLD", "oil": "USO", "usdidr": "USDIDR=X"}


def _fred_macro(start, key) -> pd.DataFrame:
    import requests
    cols = {}
    for name, sid in FRED_SERIES.items():
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": key, "file_type": "json",
                                     "observation_start": str(start.date())}, timeout=15)
            obs = r.json().get("observations", [])
            cols[name] = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                                    for o in obs if o["value"] != "."})
        except Exception:
            continue
    for name, tick in YF_MACRO.items():          # daily series FRED doesn't have (e.g. usdidr)
        if name not in cols:
            try:
                df = _yf_bars(tick, start, None, "1d")
                if df is not None and not df.empty:
                    ser = df["Close"] if "Close" in df else df["close"]
                    ser.index = pd.to_datetime(ser.index).normalize()
                    cols[name] = ser
            except Exception:
                pass
    return pd.DataFrame(cols)


def _yf_macro(start) -> pd.DataFrame:
    cols = {}
    for name, tick in YF_MACRO.items():
        df = _yf_bars(tick, start, None, "1d")
        if df is not None and not df.empty:
            ser = df["Close"] if "Close" in df else df["close"]
            ser.index = pd.to_datetime(ser.index).normalize()
            cols[name] = ser
    return pd.DataFrame(cols)


def get_macro(start, s: Settings) -> pd.DataFrame:
    df = pd.DataFrame()
    if s.fred_key:
        df = _fred_macro(start, s.fred_key)
    if df.empty:
        df = _yf_macro(start)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated()].sort_index().ffill()
    g = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["vix_lvl"] = g("vix")
    out["vix_chg5"] = g("vix").pct_change(5)
    v = g("vix")
    out["vix_z"] = (v - v.rolling(60, 20).mean()) / v.rolling(60, 20).std()
    out["y10_chg5"] = g("us10y").diff(5)
    out["yc_spread"] = g("yc_spread")
    out["spy_ret5"] = g("spy").pct_change(5)
    out["spy_ret21"] = g("spy").pct_change(21)
    out["oil_ret5"] = g("oil").pct_change(5)
    out["gold_ret5"] = g("gold").pct_change(5)
    out["dxy_chg5"] = g("dxy").pct_change(5)
    out["unrate"] = g("unrate")
    out["cons_sent"] = g("consumer_sent")
    out["usdidr_lvl"] = g("usdidr")
    out["usdidr_chg5"] = g("usdidr").pct_change(5)
    out["usdidr_chg21"] = g("usdidr").pct_change(21)
    return out


# ---------------------------------------------------------------- sentiment
# One row per (date, symbol) for every day that was fully scanned: mean article score and article
# count (news=0 / sent=NaN on quiet days). Committed to git like snapshots.csv, so the history the
# models train on grows day by day instead of being re-fetched as a short window each run.
SENT_STORE_COLS = ["date", "symbol", "sent", "news"]


def load_sentiment_store(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df[SENT_STORE_COLS]
    return pd.DataFrame(columns=SENT_STORE_COLS)


def sentiment_coverage_days(store: pd.DataFrame) -> int:
    """Days of history actually usable for training (0 if nothing was ever scored)."""
    if store.empty or not (store["news"] > 0).any():
        return 0
    return int(store["date"].nunique())


def _score_news(symbols, s: Settings, start, end, backend, max_items):
    """Score every Alpaca news item in [start, end] -> (records[date, symbol, score], first fully
    covered date). The SDK paginates internally and stops at `limit`, newest first; if it stopped
    there, the oldest day reached may be partial and is excluded."""
    from alpaca.data.requests import NewsRequest
    from .sentiment import get_scorer
    alias = {a: sym for sym in symbols for a in (sym, sym.replace("/", ""))}
    resp = _news_client(s).get_news(NewsRequest(
        symbols=",".join(symbols), start=start, end=end, sort="desc", limit=max_items,
        include_content=False))
    items = resp.data.get("news", []) if hasattr(resp, "data") else []
    texts, meta, oldest = [], [], None
    for n in items:
        day = pd.Timestamp(getattr(n, "created_at")).tz_localize(None).normalize()
        oldest = day if oldest is None else min(oldest, day)
        for sym in getattr(n, "symbols", []) or []:
            if sym in alias:
                texts.append(f"{getattr(n, 'headline', '')}. {getattr(n, 'summary', '')}")
                meta.append((day, alias[sym]))
    complete_from = pd.Timestamp(start).normalize()
    if len(items) >= max_items and oldest is not None:
        complete_from = oldest + pd.Timedelta(days=1)
    rec = pd.DataFrame({"date": [m[0] for m in meta], "symbol": [m[1] for m in meta],
                        "score": get_scorer(backend)(texts) if texts else []})
    return rec, complete_from


def update_sentiment_store(symbols, s: Settings, path: str, backend="auto", days=7,
                           start=None, end=None, max_items=3000) -> pd.DataFrame:
    """Re-scan [start, end] (default: the last `days` days) and upsert it into the store.
    Returns the store (unchanged when there are no Alpaca keys or the fetch fails)."""
    store = load_sentiment_store(path)
    if not has_alpaca(s) or not symbols:
        return store
    end = end or _utcnow()
    start = start or end - dt.timedelta(days=days)
    try:
        rec, complete_from = _score_news(symbols, s, start, end, backend, max_items)
    except Exception:
        return store
    dates = pd.date_range(complete_from, pd.Timestamp(end).normalize(), freq="D")
    if dates.empty:
        return store
    grid = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    new = (rec.groupby(["date", "symbol"])["score"].agg(sent="mean", news="count").reindex(grid)
           if not rec.empty else pd.DataFrame(index=grid, columns=["sent", "news"], dtype=float))
    new = new.reset_index()
    new["news"] = new["news"].fillna(0).astype(int)
    keep = ~(store["date"].isin(dates) & store["symbol"].isin(symbols))
    store = pd.concat([store[keep], new[SENT_STORE_COLS]], ignore_index=True)
    store = store.sort_values(["date", "symbol"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    store.to_csv(path, index=False)
    return store


def backfill_sentiment_store(symbols, s: Settings, path: str, backend="auto", days=365,
                             window_days=30, log=print) -> pd.DataFrame:
    """One-off history build (run via workflow_dispatch): scan `days` back in windows."""
    end = _utcnow()
    t = end - dt.timedelta(days=days)
    store = load_sentiment_store(path)
    while t < end:
        t2 = min(end, t + dt.timedelta(days=window_days))
        store = update_sentiment_store(symbols, s, path, backend, start=t, end=t2, max_items=200_000)
        log(f"sentiment backfill {t:%Y-%m-%d}..{t2:%Y-%m-%d}: "
            f"{sentiment_coverage_days(store)} days covered")
        t = t2
    return store


def sentiment_features(store: pd.DataFrame, symbols, end=None) -> pd.DataFrame:
    """Store -> wide daily frame of {sym}__sent (EWM of daily mean score, 0 on quiet days) and
    {sym}__news (3-day article count), the columns features._join_extras expects."""
    if store.empty:
        return pd.DataFrame()
    end = pd.Timestamp(end or _utcnow()).normalize()
    idx = pd.date_range(store["date"].min(), end, freq="D")
    base = pd.DataFrame(index=idx)
    for sym in symbols:
        d = store[store["symbol"] == sym].set_index("date")
        base[f"{sym}__sent"] = d["sent"].reindex(idx).fillna(0.0).ewm(span=3).mean()
        base[f"{sym}__news"] = d["news"].reindex(idx).fillna(0.0).rolling(3, min_periods=1).sum()
    return base


# ---------------------------------------------------------------- trading
def cancel_open_orders(tc, symbols, wait_s=15.0) -> list:
    """Cancel this universe's open orders and wait for the cancels to land. Returns orders that
    are still open afterwards. Without this, after-close orders queue until the next open, so the
    Friday, Saturday and Sunday runs each add the SAME delta on top of an unchanged position, and
    an opposite-side order is rejected as a potential wash trade."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    names = {x for sym in symbols for x in (sym, sym.replace("/", ""))}
    mine = lambda: [o for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
                    if o.symbol in names]
    orders = mine()
    for o in orders:
        try:
            tc.cancel_order_by_id(o.id)
        except Exception:
            pass
    deadline = time.monotonic() + wait_s
    while orders and time.monotonic() < deadline:
        time.sleep(1)
        orders = mine()
    return orders


def rebalance(tc, targets: dict, asset_map: dict, min_notional=20.0) -> list[dict]:
    """Move positions toward target dollar values. Any open order for a symbol being rebalanced
    is cancelled first, so the newest target replaces a queued one instead of stacking on it."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    targets = {k: v for k, v in targets.items() if v is not None and v == v}
    stuck = {o.symbol for o in cancel_open_orders(tc, list(targets))}
    positions = {p.symbol: p for p in tc.get_all_positions()}
    results = []
    for sym, tgt_val in targets.items():
        pos = positions.get(sym) or positions.get(sym.replace("/", ""))
        cur_val = float(pos.market_value) if pos else 0.0
        delta = tgt_val - cur_val
        entry = {"symbol": sym, "target": round(tgt_val, 2),
                 "current": round(cur_val, 2), "delta": round(delta, 2)}
        if sym in stuck or sym.replace("/", "") in stuck:
            results.append({**entry, "status": "skipped (open order could not be cancelled)"})
            continue
        if abs(delta) < min_notional:
            results.append({**entry, "status": "skipped (below minimum)"})
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        tif = TimeInForce.DAY if asset_map.get(sym) == "stock" else TimeInForce.GTC
        kwargs = dict(symbol=sym, side=side, time_in_force=tif)
        if side == OrderSide.SELL and pos is not None and abs(delta) >= 0.98 * abs(cur_val):
            kwargs["qty"] = abs(float(pos.qty))      # full close
        else:
            kwargs["notional"] = round(abs(delta), 2)
        try:
            o = tc.submit_order(MarketOrderRequest(**kwargs))
            results.append({**entry, "status": str(o.status), "order_id": str(o.id)})
        except Exception as e:
            results.append({**entry, "status": f"error: {e}"})
    return results
