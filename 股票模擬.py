#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股獲利計算 + 累計圖表 + 三大法人(證交所 T86) + 買賣勝率模擬

用法：
    pip install pandas numpy matplotlib requests
    python stock_tool.py --trades trades.csv
    python stock_tool.py --trades trades.csv --code 2330 --days 120 --horizon 5 --tp 0.08 --sl 0.05

trades.csv 欄位：date,code,action,price,shares
    date    : YYYY-MM-DD
    code    : 股票代號 (例如 2330)
    action  : buy / sell
    price   : 成交價
    shares  : 股數 (1張 = 1000股)
"""
import argparse
import os
import time
import warnings
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 設定
FEE_RATE = 0.001425          # 手續費牌價
TAX_STOCK = 0.003            # 一般股票證交稅 (賣出)
TAX_ETF = 0.001              # ETF 證交稅 (賣出)
MIN_FEE = 20                 # 最低手續費
SLEEP = 3.0                  # 證交所限流，請求間隔(秒)
CACHE_DIR = "inst_cache"
HEADERS = {"User-Agent": "Mozilla/5.0"}
T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
STOCK_DAY_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"


# ------------------------------------------------------------------ 網路
def get_json(url, params):
    for i in range(3):
        try:
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            except requests.exceptions.SSLError:
                r = requests.get(url, params=params, headers=HEADERS, timeout=15, verify=False)
            return r.json()
        except Exception:
            time.sleep(SLEEP * (i + 1))
    return {}


def to_num(x):
    try:
        return float(str(x).replace(",", "").replace("+", "").strip())
    except ValueError:
        return np.nan


def fetch_prices(code, start, end):
    """逐月抓證交所日 K 線資料 (開高低收量，上市股票)。
    STOCK_DAY 欄位：0日期 1成交股數 2成交金額 3開盤 4最高 5最低 6收盤 7漲跌價差 8成交筆數"""
    rows = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        js = get_json(STOCK_DAY_URL, {"date": cur.strftime("%Y%m%d"), "stockNo": code, "response": "json"})
        if js.get("stat") == "OK":
            for d in js.get("data", []):
                y, m, dd = d[0].split("/")
                rows.append({"date": pd.Timestamp(int(y) + 1911, int(m), int(dd)),
                             "open": to_num(d[3]), "high": to_num(d[4]), "low": to_num(d[5]),
                             "close": to_num(d[6]), "volume": to_num(d[1])})
        time.sleep(SLEEP)
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    df = df.dropna(subset=["close"])
    for k in ("open", "high", "low"):          # 偶爾有「--」(無成交)的欄位，用收盤價補
        df[k] = df[k].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0)
    return df


def _pick(fields, *keys):
    for i, f in enumerate(fields):
        if all(k in f for k in keys):
            return i
    return None


def fetch_t86(day):
    """抓單日三大法人買賣超 (全部個股)，有本地快取。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{day:%Y%m%d}.csv")
    cols = ["date", "code", "foreign", "trust", "total"]
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])

    js = get_json(T86_URL, {"date": f"{day:%Y%m%d}", "selectType": "ALL", "response": "json"})
    time.sleep(SLEEP)
    t = js["tables"][0] if js.get("tables") else js
    fields, data = t.get("fields"), t.get("data")
    if not data or not fields:
        empty = pd.DataFrame(columns=cols)
        if day < date.today():          # 過去的非交易日才快取空檔，今天資料可能還沒出
            empty.to_csv(path, index=False)
        return empty

    i_f = _pick(fields, "外", "買賣超")
    i_t = _pick(fields, "投信", "買賣超")
    i_a = _pick(fields, "三大法人", "買賣超")
    out = pd.DataFrame({
        "date": pd.Timestamp(day),
        "code": [str(r[0]).strip() for r in data],
        "foreign": [to_num(r[i_f]) for r in data],
        "trust": [to_num(r[i_t]) for r in data],
        "total": [to_num(r[i_a]) for r in data],
    })
    out.to_csv(path, index=False)
    return out


def fetch_institutional(days):
    frames = []
    today = date.today()
    for k in range(days, -1, -1):
        d = today - timedelta(days=k)
        if d.weekday() >= 5:
            continue
        print(f"  抓三大法人 {d} ...", end="\r")
        frames.append(fetch_t86(d))
    print(" " * 40, end="\r")
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "code", "foreign", "trust", "total"])


# ------------------------------------------------------------------ 損益計算 (FIFO)
def fees(price, shares, discount, is_sell, is_etf):
    value = price * shares
    fee = max(MIN_FEE, round(value * FEE_RATE * discount))
    tax = round(value * (TAX_ETF if is_etf else TAX_STOCK)) if is_sell else 0
    return fee, tax


def compute_pnl(trades, discount):
    lots, realized = {}, []
    for r in trades.sort_values("date").itertuples():
        code, is_etf = r.code, r.code.startswith("00")
        if r.action == "buy":
            fee, _ = fees(r.price, r.shares, discount, False, is_etf)
            lots.setdefault(code, []).append([r.shares, (r.price * r.shares + fee) / r.shares])
        else:
            fee, tax = fees(r.price, r.shares, discount, True, is_etf)
            net_per_share = (r.price * r.shares - fee - tax) / r.shares
            remain = r.shares
            q = lots.get(code, [])
            while remain > 0 and q:
                take = min(remain, q[0][0])
                realized.append({"date": r.date, "code": code, "shares": take,
                                 "pnl": (net_per_share - q[0][1]) * take})
                q[0][0] -= take
                remain -= take
                if q[0][0] == 0:
                    q.pop(0)
            if remain > 0:
                print(f"⚠ {code} 在 {r.date:%Y-%m-%d} 賣出超過持有股數 {remain} 股，已忽略該部分")
    return pd.DataFrame(realized), lots


# ------------------------------------------------------------------ 法人訊號分析 + 蒙地卡羅
def analyze(code, px, inst, args, hold=None):
    ins = inst[inst["code"] == code].set_index("date").sort_index()
    df = px.join(ins[["foreign", "trust", "total"]], how="inner").dropna()
    if len(df) < 25:
        print(f"  [{code}] 資料太少({len(df)}天)，請加大 --days")
        return None

    H = args.horizon
    df["ret1_next"] = df["close"].pct_change().shift(-1)
    df["fwd"] = df["close"].shift(-H) / df["close"] - 1
    df["sig"] = df["total"].rolling(3).sum() > 0          # 近3日三大法人合計偏買
    cur_state = bool(df["sig"].iloc[-1])

    cost = FEE_RATE * args.discount * 2 + (TAX_ETF if code.startswith("00") else TAX_STOCK)
    hist = df.dropna(subset=["fwd"])
    win_all = (hist["fwd"] > cost).mean()
    same = hist[hist["sig"] == cur_state]
    win_same = (same["fwd"] > cost).mean() if len(same) else np.nan

    label = "偏買" if cur_state else "偏賣"
    print(f"\n=== {code} 法人訊號 (近3日三大法人合計{label}) ===")
    print(f"  歷史樣本 {len(hist)} 天；買進後 {H} 日扣成本仍獲利的機率：")
    print(f"    全部日子        : {win_all:.1%}")
    print(f"    與目前同訊號({len(same)}天): {win_same:.1%}")

    # 蒙地卡羅：從「與目前同訊號」的隔日報酬中重抽樣
    pool = df[df["sig"] == cur_state]["ret1_next"].dropna().values
    if len(pool) < 20:
        pool = df["ret1_next"].dropna().values
        print("  (同訊號樣本<20，改用全部日報酬重抽樣)")
    rng = np.random.default_rng(42)
    n = args.sims
    entry = args.sim_price or px["close"].iloc[-1]
    paths = entry * np.cumprod(1 + rng.choice(pool, size=(n, H)), axis=1)
    ret_path = paths / entry - 1

    hit_tp = ret_path >= args.tp
    hit_sl = ret_path <= -args.sl
    exit_ret = ret_path[:, -1].copy()
    for i in range(n):                                    # 第一個觸發停利/停損的日子出場
        h_tp = np.argmax(hit_tp[i]) if hit_tp[i].any() else H + 1
        h_sl = np.argmax(hit_sl[i]) if hit_sl[i].any() else H + 1
        if min(h_tp, h_sl) <= H - 1:
            exit_ret[i] = ret_path[i, h_tp] if h_tp <= h_sl else ret_path[i, h_sl]
    net = exit_ret - cost

    print(f"\n--- 模擬：以 {entry:.2f} 買進，最多持有 {H} 日，停利 {args.tp:.0%} / 停損 {args.sl:.0%} ---")
    print(f"  獲利機率(扣手續費證交稅) : {(net > 0).mean():.1%}")
    print(f"  平均報酬 {net.mean():+.2%}，中位數 {np.median(net):+.2%}")
    print(f"  5%/95% 分位 : {np.percentile(net, 5):+.2%} / {np.percentile(net, 95):+.2%}")
    print(f"  觸及停利 {hit_tp.any(axis=1).mean():.1%}，觸及停損 {hit_sl.any(axis=1).mean():.1%}")

    up = (ret_path[:, -1] > 0).mean()
    print(f"\n--- 若你現在賣出 ---")
    print(f"  之後 {H} 日股價比現在高的機率 : {up:.1%}  (越高代表賣太早的可能性越大)")

    if hold:
        hold_report(code, px["close"].iloc[-1], pool, hold[0], hold[1], args)
    return df


def hold_report(code, last, pool, shares, cost_basis, args):
    """針對你『目前持有』的部位：續抱 vs 現在賣出 的模擬 (用你的實際均價與股數)。"""
    sell_rate = FEE_RATE * args.discount + (TAX_ETF if code.startswith("00") else TAX_STOCK)

    def pnl_at(price):                       # 以該價賣出，扣賣出手續費+證交稅後的總損益
        return price * shares * (1 - sell_rate) - cost_basis

    now = pnl_at(last)
    breakeven = cost_basis / (shares * (1 - sell_rate))
    days = sorted(set([1, 3, 5, 10, 20, args.horizon]))
    rng = np.random.default_rng(7)
    paths = last * np.cumprod(1 + rng.choice(pool, size=(args.sims, max(days))), axis=1)

    print(f"\n=== 你目前持有 {code}：續抱 vs 現在賣出 ===")
    print(f"  持有 {shares:,.0f} 股，均價 {cost_basis / shares:.2f}，最新 {last:.2f}，損益兩平價 {breakeven:.2f}")
    print(f"  【現在賣出】確定損益 : {now:+,.0f} 元")
    print(f"  【續抱模擬】(之後第 N 日賣出，扣成本)")
    print(f"    {'天數':>4} | {'獲利機率':>7} | {'比現在賣更好':>9} | {'平均損益':>9} | {'中位數':>9} | {'5%~95% 分位':>18}")
    for d in days:
        p = pnl_at(paths[:, d - 1])
        print(f"    {d:>4} | {(p > 0).mean():>7.1%} | {(p > now).mean():>9.1%} | "
              f"{p.mean():>+9,.0f} | {np.median(p):>+9,.0f} | {np.percentile(p, 5):>+8,.0f} ~ {np.percentile(p, 95):>+7,.0f}")
    print("  說明：『獲利機率』= 續抱到第 N 日賣出後總損益 > 0；『比現在賣更好』= 損益高於現在賣出。")
    print("  ※ 若樣本期間股價本來就在漲，結果會偏樂觀；這裡未套用最低手續費。")


def learn_backtest(code, px, inst, args):
    """逐日滾動學習：每一天只用『之前』的資料訓練，猜隔天漲/跌，再對照實際結果。"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("請先安裝：pip install scikit-learn")
        return

    ins = inst[inst["code"] == code].set_index("date").sort_index()
    df = px.join(ins[["foreign", "trust", "total"]], how="inner").dropna()
    close = df["close"]
    r = close.pct_change()

    # ---- 特徵：只用「當天收盤前」已知的資訊
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5):
        f[f"ret{k}"] = close.pct_change(k)
    f["vol5"] = r.rolling(5).std()
    f["vol20"] = r.rolling(20).std()
    f["ma5_gap"] = close / close.rolling(5).mean() - 1
    f["ma20_gap"] = close / close.rolling(20).mean() - 1
    for col in ("foreign", "trust", "total"):
        s = df[col] / (df[col].abs().rolling(20).mean() + 1)      # 用近20日平均量級標準化
        f[f"{col}_1"] = s
        f[f"{col}_3"] = s.rolling(3).sum()
        f[f"{col}_5"] = s.rolling(5).sum()
    feat_cols = list(f.columns)

    data = f.copy()
    data["y"] = (close.shift(-1) > close).astype(float)            # 1 = 隔天漲
    data.loc[data.index[-1], "y"] = np.nan                         # 最後一天還沒有答案
    valid = data.dropna(subset=feat_cols)
    labeled = valid.dropna(subset=["y"])

    min_train = 60
    n_test = min(args.test_days, len(labeled) - min_train)
    if n_test < 30:
        print(f"  [{code}] 可用資料不足 (有標籤 {len(labeled)} 天)。請加大 --days，例如 --days 450")
        return
    test_idx = labeled.index[-n_test:]

    models = {
        "邏輯迴歸": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "隨機森林": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                         random_state=0, n_jobs=-1),
    }
    preds = {k: [] for k in models}
    step = max(1, args.retrain)
    print(f"\n=== {code} 逐日滾動學習：回測最近 {n_test} 個交易日 (每 {step} 天重新訓練) ===")
    for i, t in enumerate(test_idx):
        if i % step == 0:
            train = labeled[labeled.index < t]                      # 只用 t 以前、答案已揭曉的資料
            for m in models.values():
                m.fit(train[feat_cols], train["y"])
        x = labeled.loc[[t], feat_cols]
        for k, m in models.items():
            preds[k].append(m.predict_proba(x)[0, 1])

    y = labeled.loc[test_idx, "y"].values
    rn = close.pct_change().shift(-1).loc[test_idx].values
    up_rate = y.mean()
    base = max(up_rate, 1 - up_rate)
    tc_in = FEE_RATE * args.discount
    tc_out = FEE_RATE * args.discount + (TAX_ETF if code.startswith("00") else TAX_STOCK)
    bh = np.cumprod(1 + rn) - 1

    print(f"  這段期間實際上漲天數佔 {up_rate:.1%}；『永遠猜多數方向』的基準準確率 = {base:.1%}")
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(test_idx, bh * 100, label="買進持有", color="gray", lw=2)
    for name, p in preds.items():
        p = np.array(p)
        guess = (p > 0.5).astype(int)
        hit = guess == y
        acc = hit.mean()
        z = (acc - 0.5) / np.sqrt(0.25 / len(y))
        pos = guess.astype(float)
        prev = np.concatenate([[0.0], pos[:-1]])
        cost = np.where((pos == 1) & (prev == 0), tc_in, 0) + np.where((pos == 0) & (prev == 1), tc_out, 0)
        cum = np.cumprod(1 + pos * rn - cost) - 1
        up_g, dn_g = guess == 1, guess == 0
        print(f"\n  [{name}]")
        print(f"    猜對率 {acc:.1%}  (基準 {base:.1%}；對 50% 的 z 值 {z:+.2f}，|z|>2 才算比亂猜明顯好)")
        print(f"    猜漲 {up_g.sum()} 次，猜漲時猜對 {hit[up_g].mean() if up_g.any() else float('nan'):.1%}；"
              f"猜跌 {dn_g.sum()} 次，猜跌時猜對 {hit[dn_g].mean() if dn_g.any() else float('nan'):.1%}")
        print(f"    照訊號操作 (猜漲持有、猜跌空手、扣成本) 累計 {cum[-1]:+.1%}，買進持有 {bh[-1]:+.1%}")
        ax[0].plot(test_idx, cum * 100, label=f"{name} 訊號操作")
        ax[1].plot(test_idx, pd.Series(hit.astype(float)).rolling(30).mean().values * 100, label=f"{name} 近30日猜對率")
    ax[1].axhline(50, color="gray", ls="--", lw=0.8)
    ax[0].set_title(f"{code} 逐日滾動學習回測")
    ax[0].set_ylabel("累計報酬 %")
    ax[1].set_ylabel("猜對率 %")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"learn_{code}.png", dpi=150)
    print(f"\n已輸出 learn_{code}.png")

    # ---- 用全部歷史資料訓練，猜『下一個交易日』
    if valid.index[-1] == data.index[-1]:
        x_last = valid.loc[[valid.index[-1]], feat_cols]
        print(f"\n  --- 明日預測 (以 {valid.index[-1]:%Y-%m-%d} 收盤資料為準) ---")
        for name, m in models.items():
            m.fit(labeled[feat_cols], labeled["y"])
            print(f"    {name}：隔天上漲機率 {m.predict_proba(x_last)[0, 1]:.1%}")
        imp = pd.Series(models["隨機森林"].feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("    隨機森林認為最重要的特徵：" + "、".join(f"{k}({v:.2f})" for k, v in imp.head(5).items()))
    print("  ※ 短線漲跌接近隨機，猜對率通常只在 50%~55%，請以 z 值與樣本數判斷是否真有效。")


# ------------------------------------------------------------------ 圖表
def plot_cumulative(realized, unrealized_total, path="pnl_cumulative.png"):
    if realized.empty:
        print("沒有已實現損益可畫圖。")
        return
    r = realized.copy()
    r["date"] = pd.to_datetime(r["date"])
    daily = r.groupby("date")["pnl"].sum()
    cum = daily.cumsum()

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(cum.index, cum.values, marker="o", lw=2)
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_title(f"累計已實現損益 (含未實現共 {realized['pnl'].sum() + unrealized_total:,.0f} 元)")
    ax[0].set_ylabel("NT$")
    ax[0].grid(alpha=0.3)
    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in daily.values]   # 台股：紅漲綠跌
    ax[1].bar(daily.index, daily.values, color=colors)
    ax[1].set_ylabel("單日已實現損益")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"已輸出 {path}")


def plot_inst(code, df):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(df.index, df["close"], lw=2)
    ax[0].set_title(f"{code} 收盤價與三大法人買賣超 (張)")
    ax[0].grid(alpha=0.3)
    ax[1].bar(df.index, df["total"] / 1000, label="三大法人合計", alpha=0.5, color="gray")
    ax[1].plot(df.index, (df["foreign"] / 1000).cumsum() - (df["foreign"] / 1000).cumsum().iloc[0], label="外資累計")
    ax[1].plot(df.index, (df["trust"] / 1000).cumsum() - (df["trust"] / 1000).cumsum().iloc[0], label="投信累計")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"inst_{code}.png", dpi=150)
    print(f"已輸出 inst_{code}.png")

def kline_report(code, px):
    """K 線判讀 + 技術指標現況 + K 線圖 (輸出 kline_{code}.png)。"""
    if px is None or len(px) < 20:
        print(f"  [{code}] K 線資料不足，略過技術指標判讀。")
        return

    df = px.copy()
    close = df["close"]
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))

    last_idx = df.index[-1]
    c_last = close.iloc[-1]
    o_last = df["open"].iloc[-1]
    h_last = df["high"].iloc[-1]
    l_last = df["low"].iloc[-1]

    # 簡易單根 K 線型態判讀
    body = abs(c_last - o_last)
    rng = max(h_last - l_last, 1e-9)
    lower_shadow = min(o_last, c_last) - l_last
    upper_shadow = h_last - max(o_last, c_last)

    if c_last > o_last:
        candle_desc = "紅 K（收高於開）"
    elif c_last < o_last:
        candle_desc = "黑 K（收低於開）"
    else:
        candle_desc = "十字／平盤 K"

    if lower_shadow > body * 2 and upper_shadow < body:
        candle_desc += "、下影線較長（低檔有承接）"
    elif upper_shadow > body * 2 and lower_shadow < body:
        candle_desc += "、上影線較長（高檔有壓）"

    ma60_str = f"{ma60.iloc[-1]:.2f}" if pd.notna(ma60.iloc[-1]) else "資料不足"
    rsi_str = f"{rsi14.iloc[-1]:.1f}" if pd.notna(rsi14.iloc[-1]) else "N/A"

    print(f"\n=== {code} K 線與技術指標現況 ({last_idx:%Y-%m-%d}) ===")
    print(f"  最新 OHLC : 開 {o_last:.2f} / 高 {h_last:.2f} / 低 {l_last:.2f} / 收 {c_last:.2f}  [{candle_desc}]")
    print(f"  均線位置  : MA5={ma5.iloc[-1]:.2f} | MA20={ma20.iloc[-1]:.2f} | MA60={ma60_str}")
    print(f"  RSI(14)   : {rsi_str}")

    # 畫近 90 個交易日的 K 線 + 均線 + 成交量
    sub = df.tail(90).copy()
    s_ma5 = ma5.reindex(sub.index)
    s_ma20 = ma20.reindex(sub.index)
    s_ma60 = ma60.reindex(sub.index)

    x = np.arange(len(sub))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    for i, (_, row) in enumerate(sub.iterrows()):
        is_up = row["close"] >= row["open"]
        col = "#d62728" if is_up else "#2ca02c"          # 台股紅漲綠跌
        ax[0].vlines(i, row["low"], row["high"], color=col, lw=1.0)
        b_bot = min(row["open"], row["close"])
        b_h = max(abs(row["close"] - row["open"]), 0.05)
        ax[0].add_patch(plt.Rectangle((i - 0.3, b_bot), 0.6, b_h, facecolor=col, edgecolor=col, alpha=0.85))

    ax[0].plot(x, s_ma5.values, label="MA5", lw=1.2)
    ax[0].plot(x, s_ma20.values, label="MA20", lw=1.4)
    if s_ma60.notna().any():
        ax[0].plot(x, s_ma60.values, label="MA60", lw=1.4)
    ax[0].set_title(f"{code} 日 K 線與均線")
    ax[0].set_ylabel("價格")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)

    vol_colors = ["#d62728" if r["close"] >= r["open"] else "#2ca02c" for _, r in sub.iterrows()]
    ax[1].bar(x, sub["volume"].values / 1000, color=vol_colors, alpha=0.75)
    ax[1].set_ylabel("成交量 (張)")
    ax[1].grid(alpha=0.3)

    step_ticks = max(1, len(sub) // 10)
    tick_pos = list(range(0, len(sub), step_ticks))
    ax[1].set_xticks(tick_pos)
    ax[1].set_xticklabels([sub.index[j].strftime("%m-%d") for j in tick_pos], rotation=30)

    fig.tight_layout()
    out_path = f"kline_{code}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已輸出 {out_path}")
# ------------------------------------------------------------------ 主程式
def main():
    # 不管從哪個資料夾執行，都切到腳本所在資料夾 (trades.csv、快取、圖表都放這裡)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="trades.csv")
    ap.add_argument("--code", help="只分析這檔 (預設分析目前仍持有的所有股票)")
    ap.add_argument("--discount", type=float, default=0.6, help="手續費折扣，0.6 = 六折")
    ap.add_argument("--days", type=int, default=450, help="三大法人回溯日曆天數 (越長越準，但第一次抓越久)")
    ap.add_argument("--horizon", type=int, default=5, help="模擬持有交易日數")
    ap.add_argument("--tp", type=float, default=0.08, help="停利比例")
    ap.add_argument("--sl", type=float, default=0.05, help="停損比例")
    ap.add_argument("--sim-price", type=float, help="模擬進場價 (預設最新收盤)")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--learn", action=argparse.BooleanOptionalAction, default=True,
                    help="逐日滾動學習回測 (預設開啟；不想跑請加 --no-learn)")
    ap.add_argument("--test-days", type=int, default=180, help="回測最近幾個交易日")
    ap.add_argument("--retrain", type=int, default=5, help="每幾天重新訓練一次模型")
    args = ap.parse_args()

    trades = pd.read_csv(args.trades, dtype={"code": str}, encoding="utf-8-sig")
    trades.columns = [c.strip().lower() for c in trades.columns]
    trades = trades.dropna(how="all")                                  # 刪掉完全空白的列
    bad = trades[trades[["date", "code", "action", "price", "shares"]].isna().any(axis=1)]
    if len(bad):
        print("⚠ 以下列有缺漏欄位，已略過 (行號 = Excel 的列號)：")
        print((bad.index + 2).tolist())
        trades = trades.drop(bad.index)
    trades["date"] = pd.to_datetime(trades["date"])
    trades["code"] = trades["code"].astype(str).str.strip()
    trades["action"] = trades["action"].astype(str).str.lower().str.strip()
    trades["price"] = pd.to_numeric(trades["price"])
    trades["shares"] = pd.to_numeric(trades["shares"])
    if not trades["action"].isin(["buy", "sell"]).all():
        raise SystemExit("action 欄只能填 buy 或 sell，請檢查 trades.csv")
    realized, lots = compute_pnl(trades, args.discount)

    codes = [args.code] if args.code else [c for c, q in lots.items() if sum(x[0] for x in q) > 0]
    if not codes:
        codes = sorted(trades["code"].unique())

    start = date.today() - timedelta(days=args.days + 120)       # 多抓 120 天給 60 日均線等指標暖機
    prices = {}
    print("抓取股價...")
    for c in codes:
        prices[c] = fetch_prices(c, start, date.today())

    # 損益彙總
    print("\n================ 損益彙總 ================")
    if not realized.empty:
        print(realized.groupby("code")["pnl"].sum().round(0).to_string())
    print(f"已實現合計：{realized['pnl'].sum() if not realized.empty else 0:,.0f} 元")
    unrealized_total = 0.0
    for c, q in lots.items():
        sh = sum(x[0] for x in q)
        if sh <= 0 or c not in prices or prices[c].empty:
            continue
        last = prices[c]["close"].iloc[-1]
        cost_basis = sum(x[0] * x[1] for x in q)
        u = last * sh - cost_basis - last * sh * (FEE_RATE * args.discount + (TAX_ETF if c.startswith("00") else TAX_STOCK))
        unrealized_total += u
        print(f"持有 {c}: {sh:,.0f} 股，均價 {cost_basis / sh:.2f}，最新 {last:.2f}，未實現(扣賣出成本) {u:+,.0f}")
    print(f"未實現合計：{unrealized_total:+,.0f} 元")
    plot_cumulative(realized, unrealized_total)

    # 三大法人 + 模擬
    print("\n抓取三大法人資料 (第一次較久，之後走快取)...")
    inst = fetch_institutional(args.days)
    if inst.empty:
        print("抓不到三大法人資料，請確認網路連線。")
        return
    for c in codes:
        if prices[c].empty:
            print(f"[{c}] 抓不到股價 (此腳本僅支援上市股票)")
            continue
        q = lots.get(c, [])
        sh = sum(x[0] for x in q)
        hold = (sh, sum(x[0] * x[1] for x in q)) if sh > 0 else None
        df = analyze(c, prices[c], inst, args, hold)
        if df is not None:
            plot_inst(c, df)
        kline_report(c, prices[c])                       # K 線判讀 + 技術指標現況 + K 線圖
        if args.learn:                                   # 接著跑逐日滾動學習 (K線+指標+法人 結合)
            learn_backtest(c, prices[c], inst, args)

    print("\n※ 模擬只是用歷史統計做機率估計，不代表未來，也不是投資建議。")


if __name__ == "__main__":
    main()
