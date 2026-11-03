#!/usr/bin/env python3
"""
Train a small, auditable 1-minute Forex direction model from Twelve Data candles.
This is research code, not a profitable strategy guarantee.
Install: pip install requests pandas numpy scikit-learn
Run: python train_model.py --symbol EUR/USD --apikey YOUR_KEY
"""
import argparse, json, math, os, sys
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, log_loss

FEATURES = ["ret1", "ret3", "ret5", "ema_gap", "rsi14", "range_pct", "body_pct", "volatility10"]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def fetch(symbol, apikey, outputsize):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1min", "outputsize": outputsize,
              "apikey": apikey, "format": "JSON", "timezone": "UTC"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") == "error" or "values" not in payload:
        raise RuntimeError(payload.get("message", str(payload)))
    df = pd.DataFrame(payload["values"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    return df

def make_features(df):
    c, h, l, o = df.close, df.high, df.low, df.open
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change(1)
    f["ret3"] = c.pct_change(3)
    f["ret5"] = c.pct_change(5)
    ema_fast = c.ewm(span=8, adjust=False).mean()
    ema_slow = c.ewm(span=21, adjust=False).mean()
    f["ema_gap"] = (ema_fast - ema_slow) / c
    f["rsi14"] = rsi(c) / 100.0
    f["range_pct"] = (h - l) / c
    f["body_pct"] = (c - o) / c
    f["volatility10"] = c.pct_change().rolling(10).std()
    # Predict next candle direction, not the current candle.
    y = (c.shift(-1) > c).astype(int).rename("target")
    data = f.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--apikey", default=os.getenv("TWELVE_DATA_API_KEY"))
    ap.add_argument("--outputsize", type=int, default=5000)
    ap.add_argument("--out", default="../app/src/main/assets/model.json")
    args = ap.parse_args()
    if not args.apikey:
        sys.exit("API key missing. Use --apikey or TWELVE_DATA_API_KEY environment variable.")
    raw = fetch(args.symbol, args.apikey, args.outputsize)
    data = make_features(raw)
    if len(data) < 300:
        sys.exit(f"Only {len(data)} usable rows. Need at least 300; increase API history/plan.")
    X = data[FEATURES].to_numpy(dtype=float)
    y = data.target.to_numpy(dtype=int)

    # Chronological split; do not shuffle time-series data.
    split = int(len(data) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    pred = (probs >= 0.5).astype(int)

    report = {
        "symbol": args.symbol, "timeframe": "1min", "samples": int(len(data)),
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_up": float(precision_score(y_test, pred, zero_division=0)),
        "recall_up": float(recall_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "features": FEATURES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "thresholds": {"buy": 0.60, "sell": 0.40},
        "warning": "Research model only. Historical test metrics do not predict future results."
    }
    if len(np.unique(y_test)) == 2:
        report["log_loss"] = float(log_loss(y_test, probs))
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("mean","scale","coef","intercept")}, indent=2))
    print(f"\nModel artifact saved: {out_path}")
    print("Do not deploy unless out-of-sample results are stable across multiple pairs and date ranges.")

if __name__ == "__main__":
    main()
