#!/usr/bin/env python3
"""
bot.py - runs ONCE per hour, called by GitHub Actions. Then exits.

SIMULATION ONLY. There is no exchange account, no exchange API key, and no
order function anywhere in this file. It reads public price data that needs
no login. Nothing here can spend money.

THE EXPERIMENT
  10 strategies. Each runs TWICE as an identical twin (A and B).
  Same prompt, same rules, same prices. They differ only because the AI
  answers the same question slightly differently each time.

  The gap between twins IS the noise floor - measured, not guessed.
  If TREND-A and TREND-B finish 12% apart, nothing means anything until
  it beats 12%. That number is on the dashboard.

RULES
  One BUY per book per day. Selling is unlimited.
  Long or flat only. No shorting, no leverage, no margin.
  Position size set by a 1% risk rule the AI cannot override.

CONTROLS (do not delete)
  RANDOM - coin flip
  HODL   - buy once, never sell
  CROSS  - moving-average crossover, no AI at all. It may well win.
"""

import os, json, time, random, statistics
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests

# ───────────────────────────── settings ─────────────────────────────

SYMBOL       = "ETH/USD"
TIMEFRAME    = "1h"
CAPITAL      = 100.0
RISK_PCT     = 0.01
FEE_RATE     = 0.001
SLIPPAGE     = 0.0005
ATR_LEN      = 14
STOP_MULT    = 2.5
WINDOW       = 48
MIN_NOTIONAL = 5.0
PHASE_A_BARS = 720           # ~30 days of hourly bars
PHASE_B_BARS = 720
HISTORY_MAX  = 800

# ─── swap the model here. Only place the provider appears. ───
PROVIDER    = "gemini"                 # "gemini" or "anthropic"
GEMINI_M    = "gemini-2.5-flash"       # free tier. also try gemini-3-flash-preview
ANTHROPIC_M = "claude-sonnet-4-6"      # paid, if you ever switch
CALL_GAP    = 5.0                      # seconds between calls (free tier ~10-15/min)

STATE = Path("state.json")
DATA  = Path("docs/data.json")

# ─── 10 strategies. Everything identical except one attention hint. ───
STRATEGIES = [
    ("BASE",   ""),   # control: no hint. If nothing beats this, hints do nothing.
    ("TREND",  "Pay attention to the overall direction of price across the whole window."),
    ("LEVELS", "Pay attention to prices where the market has turned around before."),
    ("SWING",  "Pay attention to how much price is moving now versus earlier in the window."),
    ("RECENT", "Weight the last few candles more heavily than the older ones."),
    ("SHAPE",  "Pay attention to the shape of the price path, not just where it ended."),
    ("MEAN",   "Pay attention to how far price is from its average over the window."),
    ("RANGE",  "Pay attention to where price sits between the window's high and low."),
    ("SPEED",  "Pay attention to how fast price is moving, not only which direction."),
    ("CALM",   "Prefer to act when the market is quiet rather than when it is jumpy."),
]
TWINS = ["A", "B"]
BOOKS = [f"{n}-{t}" for n, _ in STRATEGIES for t in TWINS]
HINT  = {n: h for n, h in STRATEGIES}

# ───────────────────────────── state ────────────────────────────────

def blank(name, kind="ai"):
    return dict(name=name, kind=kind, cash=CAPITAL, pos=None, trades=[],
                peak=CAPITAL, last_buy="", phaseA_end=None, confs=[])

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    books = {b: blank(b) for b in BOOKS}
    for c in ("RANDOM", "HODL", "CROSS"):
        books[c] = blank(c, kind="ctrl")
    return dict(bar=0, last_ts=None, phase="A", winner=None, done=False,
                calls=0, calls_today=0, day="", fails=0, books=books,
                history=[], recent=[])

def atr(c, n=ATR_LEN):
    tr = [max(c[i][2]-c[i][3], abs(c[i][2]-c[i-1][4]), abs(c[i][3]-c[i-1][4]))
          for i in range(1, len(c))]
    return statistics.fmean(tr[-n:]) if tr else c[-1][4]*0.01

def sma(c, n):
    return statistics.fmean([x[4] for x in c[-n:]])

def eq(b, px):
    return b["cash"] + (b["pos"]["qty"]*px if b["pos"] else 0.0)

# ────────────────────────── paper execution ─────────────────────────

def buy(b, px, stop, day, conf=None):
    """Risk rule sizes the trade, then no-leverage caps it. AI cannot override."""
    if b["last_buy"] == day:
        return False                       # one buy per day. Hard rule.
    e, dist = eq(b, px), px - stop
    if dist <= 0:
        return False
    qty = min((e * RISK_PCT) / dist, e / px)
    if qty * px < MIN_NOTIONAL:
        return False
    fill = px * (1 + SLIPPAGE)
    cost = qty * fill
    fee  = cost * FEE_RATE
    if cost + fee > b["cash"]:
        return False
    b["cash"] -= cost + fee
    b["pos"]   = dict(qty=qty, entry=fill, stop=stop, risk=qty*dist, conf=conf)
    b["last_buy"] = day
    return True

def sell(b, px):
    p = b["pos"]
    if not p:
        return 0.0
    fill  = px * (1 - SLIPPAGE)
    gross = p["qty"] * fill
    fee   = gross * FEE_RATE
    b["cash"] += gross - fee
    pnl = (fill - p["entry"]) * p["qty"] - fee
    b["trades"].append(dict(pnl=round(pnl, 4),
                            R=round(pnl / p["risk"], 3) if p["risk"] else 0))
    if p.get("conf") is not None:
        b["confs"].append(dict(conf=p["conf"], won=pnl > 0))
    b["pos"] = None
    return pnl

# ─────────────────────────── model call ─────────────────────────────

def call_model(prompt):
    """Returns a parsed dict. Raises on any failure - the caller counts it."""
    if PROVIDER == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_M}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"],
                     "content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 400, "temperature": 1.0,
                                       "responseMimeType": "application/json",
                                       "thinkingConfig": {"thinkingBudget": 0}}},
            timeout=45)
       if r.status_code != 200:
            if r.status_code == 429:
                time.sleep(20)             # free-tier rate limit, back off
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_M, "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=45)
        r.raise_for_status()
        txt = "".join(x.get("text", "") for x in r.json()["content"])

    txt = txt.replace("```json", "").replace("```", "").strip()
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])

def build_prompt(strat, b, candles, a, can_buy):
    closes = ",".join(f"{c[4]:.2f}" for c in candles[-WINDOW:])
    px = candles[-1][4]
    p  = b["pos"]
    if p:
        pos  = f"HOLDING since {p['entry']:.2f}. Stop at {p['stop']:.2f}."
        opts = '"hold" or "close"'
    else:
        pos  = "FLAT (no position)."
        opts = '"hold" or "buy"' if can_buy else '"hold" (no buy left today)'
    return f"""You trade spot {SYMBOL}. You can only be long or flat. No shorting, no leverage.
{HINT[strat]}

Last {WINDOW} hourly closes, oldest first: {closes}
Current price: {px:.2f}
Typical hourly move (ATR{ATR_LEN}): {a:.2f}
Your position: {pos}
You may buy at most once per day.

Position size is decided for you by a fixed risk rule. Do not choose size.

Reply with ONLY this JSON object, nothing else:
{{"action": {opts}, "confidence": <0-100>, "reason": "<max 10 words>"}}"""

# ──────────────────────────── one bar ───────────────────────────────

def step(s, candles):
    ts  = datetime.fromtimestamp(candles[-1][0] / 1000, timezone.utc)
    day = ts.strftime("%Y-%m-%d")
    iso = ts.isoformat(timespec="minutes")
    px, low = candles[-1][4], candles[-1][3]
    a = atr(candles)
    books = s["books"]
    random.seed(20260804 + s["bar"])

    if s["day"] != day:
        s["day"], s["calls_today"] = day, 0

    def note(book, act, why):
        s["recent"].insert(0, dict(t=iso[5:], book=book, act=act, why=why))

    # stops
    for b in books.values():
        if b["pos"] and b["name"] != "HODL" and low <= b["pos"]["stop"]:
            sp = b["pos"]["stop"]
            pnl = sell(b, sp)
            note(b["name"], "STOP", f"{sp:.2f} ({pnl:+.2f})")

    # HODL
    h = books["HODL"]
    if not h["pos"] and h["cash"] > MIN_NOTIONAL:
        fill = px * (1 + SLIPPAGE)
        qty  = h["cash"] / (fill * (1 + FEE_RATE))
        h["cash"] -= qty * fill * (1 + FEE_RATE)
        h["pos"] = dict(qty=qty, entry=fill, stop=0.0, risk=1e9)

    # CROSS - no AI, free to run, and it might beat everything
    cb = books["CROSS"]
    if len(candles) >= 30:
        fast, slow = sma(candles, 10), sma(candles, 30)
        if fast > slow and not cb["pos"]:
            if buy(cb, px, px - STOP_MULT * a, day):
                note("CROSS", "BUY", "10 crossed above 30")
        elif fast < slow and cb["pos"]:
            pnl = sell(cb, px)
            note("CROSS", "CLOSE", f"crossed back ({pnl:+.2f})")

    # the 20 AI books
    active = [s["winner"]] if (s["phase"] == "B" and s["winner"]) else BOOKS
    entered = False
    for name in active:
        b = books[name]
        strat = name.rsplit("-", 1)[0]
        can_buy = b["last_buy"] != day

        # quota saver: flat with no buy left means nothing can happen. Skip the call.
        if not b["pos"] and not can_buy:
            continue

        try:
            d = call_model(build_prompt(strat, b, candles, a, can_buy))
            s["calls"] += 1
            s["calls_today"] += 1
        except Exception as e:
            s["fails"] += 1
            note(name, "FAIL", str(e)[:70])
            continue
        time.sleep(CALL_GAP)

        act  = str(d.get("action", "hold")).lower().strip()
        why  = str(d.get("reason", ""))[:60]
        conf = d.get("confidence")
        conf = conf if isinstance(conf, (int, float)) else None

        if act == "buy" and not b["pos"]:
            if not can_buy:
                s["fails"] += 1                    # tried to break the daily rule
                note(name, "BLOCK", "already bought today")
            elif buy(b, px, px - STOP_MULT * a, day, conf):
                entered = True
                note(name, "BUY", why)
        elif act == "close" and b["pos"]:
            pnl = sell(b, px)
            note(name, "CLOSE", f"{why} ({pnl:+.2f})")
        elif act not in ("buy", "hold", "close"):
            s["fails"] += 1
            note(name, "FAIL", f"bad action '{act[:12]}'")

    # RANDOM mirrors entry timing, flips a coin
    rb = books["RANDOM"]
    if entered and not rb["pos"] and random.random() < 0.5:
        buy(rb, px, px - STOP_MULT * a, day)
    elif rb["pos"] and random.random() < 0.04:
        sell(rb, px)

    for b in books.values():
        b["peak"] = max(b["peak"], eq(b, px))

    s["history"].append(dict(t=iso, px=round(px, 2),
                             eq={n: round(eq(b, px), 2) for n, b in books.items()}))
    s["history"] = s["history"][-HISTORY_MAX:]
    s["recent"]  = s["recent"][:30]
    s["bar"] += 1

    if s["phase"] == "A" and s["bar"] >= PHASE_A_BARS:
        s["winner"] = max(BOOKS, key=lambda n: eq(books[n], px))
        s["phase"]  = "B"
        for b in books.values():
            b["phaseA_end"] = round(eq(b, px), 2)
        note("SYSTEM", "PHASE", f"A over. Leader {s['winner']}. Re-testing it alone.")
    elif s["phase"] == "B" and s["bar"] >= PHASE_A_BARS + PHASE_B_BARS:
        s["done"] = True

# ─────────────────────────── dashboard data ─────────────────────────

def emit(s, px):
    books = s["books"]
    rows = []
    for n, b in books.items():
        tr, e = b["trades"], eq(b, px)
        rows.append(dict(
            name=n, kind=b["kind"], eq=round(e, 2),
            ret=round((e / CAPITAL - 1) * 100, 2), n=len(tr),
            win=round(100 * sum(1 for t in tr if t["pnl"] > 0) / len(tr)) if tr else 0,
            avgR=round(statistics.fmean([t["R"] for t in tr]), 2) if tr else 0,
            dd=round((b["peak"] - e) / b["peak"] * 100, 1) if b["peak"] else 0,
            inpos=bool(b["pos"])))
    rows.sort(key=lambda x: -x["eq"])

    # THE measurement: how far apart do identical twins finish?
    gaps = []
    for name, _ in STRATEGIES:
        A, B = books.get(f"{name}-A"), books.get(f"{name}-B")
        if A and B:
            ea, eb = eq(A, px), eq(B, px)
            gaps.append(dict(s=name, a=round(ea, 2), b=round(eb, 2),
                             gap=round(abs(ea - eb) / CAPITAL * 100, 2)))
    floor = round(statistics.median([g["gap"] for g in gaps]), 2) if gaps else 0

    calib = []
    allc = [c for n in BOOKS for c in books[n]["confs"]]
    for lo, hi in ((0, 50), (50, 65), (65, 80), (80, 101)):
        g = [c for c in allc if lo <= c["conf"] < hi]
        if len(g) >= 3:
            calib.append(dict(band=f"{lo}-{hi}", n=len(g),
                              hit=round(100 * sum(1 for c in g if c["won"]) / len(g))))

    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(dict(
        updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        symbol=SYMBOL, timeframe=TIMEFRAME, price=round(px, 2), capital=CAPITAL,
        bar=s["bar"], phase=s["phase"], winner=s["winner"], done=s["done"],
        phaseA=PHASE_A_BARS, phaseB=PHASE_B_BARS,
        provider=PROVIDER, model=GEMINI_M if PROVIDER == "gemini" else ANTHROPIC_M,
        calls=s["calls"], today=s["calls_today"], fails=s["fails"],
        floor=floor, twins=gaps, books=rows, calib=calib,
        history=[dict(t=h["t"][5:], px=h["px"], eq=h["eq"]) for h in s["history"]],
        recent=s["recent"][:25]), separators=(",", ":")))
EXCHANGES = ["kraken", "coinbaseexchange", "bitstamp"]

def fetch_candles():
    """GitHub's runners are in the US and Binance returns 451 there.
    Try exchanges in order until one answers."""
    last = None
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            c = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=WINDOW + ATR_LEN + 5)
            if c and len(c) > ATR_LEN + 5:
                print(f"prices from {name}")
                return c[:-1]
        except Exception as e:
            last = f"{name}: {type(e).__name__}"
    raise RuntimeError(f"no exchange reachable ({last})")

def main():
    s = load()
    if s.get("done"):
        print("experiment complete")
        return
    c = fetch_candles()
    if c[-1][0] == s["last_ts"]:
        print("no new candle, refreshing dashboard only")
        emit(s, c[-1][4])
        return
    s["last_ts"] = c[-1][0]
    step(s, c)
    STATE.write_text(json.dumps(s, separators=(",", ":")))
    emit(s, c[-1][4])
    print(f"bar {s['bar']} phase {s['phase']} px {c[-1][4]:.2f} "
          f"calls {s['calls_today']} today, fails {s['fails']}")

if __name__ == "__main__":
    main()
