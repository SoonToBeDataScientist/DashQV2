from __future__ import annotations

import datetime as dt

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


def get_daily_bars(symbols, asset_map, start, end, s: Settings) -> dict:
    out = {}
    stocks = [x for x in symbols if asset_map.get(x) == "stock"]
    cryptos = [x for x in symbols if asset_map.get(x) == "crypto"]
    if has_alpaca(s):
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            if stocks:
                req = StockBarsRequest(symbol_or_symbols=stocks, timeframe=TimeFrame.Day,
                                       start=start, end=end, feed=DataFeed.IEX)
                df = _stock_client(s).get_stock_bars(req).df
                for sym in stocks:
                    try:
                        out[sym] = _clean_bars(df.xs(sym, level="symbol"), daily=True)
                    except KeyError:
                        pass
            if cryptos:
                req = CryptoBarsRequest(symbol_or_symbols=cryptos, timeframe=TimeFrame.Day,
                                        start=start, end=end)
                df = _crypto_client(s).get_crypto_bars(req).df
                for sym in cryptos:
                    try:
                        out[sym] = _clean_bars(df.xs(sym, level="symbol"), daily=True)
                    except KeyError:
                        pass
        except Exception:
            pass
    for sym in symbols:  # yfinance fallback for anything missing
        if sym not in out:
            df = _yf_bars(sym, start, end, "1d")
            if df is not None and not df.empty:
                out[sym] = _clean_bars(df, daily=True)
    return out


def get_intraday_bars(sym, asset_class, s: Settings, minutes=15, days=3) -> pd.DataFrame:
    end = dt.datetime.utcnow()
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
    df = _yf_bars(sym, dt.datetime.utcnow() - dt.timedelta(days=7), None, "1d")
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else float("nan")


# ---------------------------------------------------------------- macro
FRED_SERIES = {"vix": "VIXCLS", "us10y": "DGS10", "yc_spread": "T10Y2Y",
               "unrate": "UNRATE", "consumer_sent": "UMCSENT", "oil": "DCOILWTICO"}
YF_MACRO = {"vix": "^VIX", "us10y": "^TNX", "dxy": "DX-Y.NYB",
            "spy": "SPY", "gold": "GLD", "oil": "USO"}


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
    return out


# ---------------------------------------------------------------- sentiment
def get_news_sentiment(symbols, s: Settings, days=45, max_pages=4, backend="auto") -> pd.DataFrame:
    """Daily sentiment (VADER or FinBERT) + news counts per symbol."""
    today = dt.datetime.utcnow().date()
    idx = pd.date_range(today - dt.timedelta(days=days), today, freq="D")
    base = pd.DataFrame(index=idx)
    for sym in symbols:
        base[f"{sym}__sent"], base[f"{sym}__news"] = 0.0, 0.0
    if not has_alpaca(s):
        return base
    try:
        from alpaca.data.requests import NewsRequest
        from .sentiment import get_scorer
        scorer = get_scorer(backend)
        alias = {sym: {sym, sym.replace("/", "")} for sym in symbols}
        texts, meta, token = [], [], None
        for _ in range(max_pages):
            kwargs = dict(symbols=",".join(symbols),
                          start=dt.datetime.utcnow() - dt.timedelta(days=days),
                          limit=50, include_content=False)
            if token:
                kwargs["page_token"] = token
            resp = _news_client(s).get_news(NewsRequest(**kwargs))
            items = resp.data.get("news", []) if hasattr(resp, "data") else []
            for n in items:
                text = f"{getattr(n, 'headline', '')}. {getattr(n, 'summary', '')}"
                day = pd.Timestamp(getattr(n, "created_at")).tz_localize(None).normalize()
                for sym in getattr(n, "symbols", []) or []:
                    for orig, aliases in alias.items():
                        if sym in aliases:
                            texts.append(text)
                            meta.append((day, orig))
            token = getattr(resp, "next_page_token", None)
            if not token or not items:
                break
        if not texts:
            return base
        rec = pd.DataFrame({"date": [m[0] for m in meta], "symbol": [m[1] for m in meta],
                            "score": scorer(texts)})
        mean = rec.pivot_table(index="date", columns="symbol", values="score", aggfunc="mean")
        cnt = rec.pivot_table(index="date", columns="symbol", values="score", aggfunc="count")
        for sym in symbols:
            if sym in mean:
                base[f"{sym}__sent"] = mean[sym].reindex(idx).fillna(0).ewm(span=3).mean()
                base[f"{sym}__news"] = cnt[sym].reindex(idx).fillna(0).rolling(3, min_periods=1).sum()
        return base
    except Exception:
        return base


# ---------------------------------------------------------------- trading
def rebalance(tc, targets: dict, asset_map: dict, min_notional=20.0) -> list[dict]:
    """Submit market orders to move positions toward target dollar values."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    positions = {p.symbol: p for p in tc.get_all_positions()}
    results = []
    for sym, tgt_val in targets.items():
        pos = positions.get(sym) or positions.get(sym.replace("/", ""))
        cur_val = float(pos.market_value) if pos else 0.0
        delta = tgt_val - cur_val
        entry = {"symbol": sym, "target": round(tgt_val, 2),
                 "current": round(cur_val, 2), "delta": round(delta, 2)}
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
