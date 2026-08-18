#!/usr/bin/env python3
"""
bot.py - runs ONCE per hour, called by GitHub Actions. Then exits.

SIMULATION ONLY. No exchange account, no exchange API key, no order function.
It reads public price data that needs no login. Nothing here can spend money.

EVERYTHING IS IN EUROS. Market is ETH/EUR, so no conversion anywhere.

FAILURE ARCHIVING
  Bump VERSION below whenever you change anything. On the next run the bot
  appends every accumulated failure to docs/fails.csv, clears them out of the
  live log, and resets the counter to zero. The dashboard then shows only
  failures caused by the CURRENT version. Nothing is deleted - the CSV keeps
  the full history, and the lifetime total is still reported separately.

THE EXPERIMENT
  10 strategies, each run TWICE as identical twins (A and B).
  The gap between twins IS the noise floor - measured, not guessed.

RULES
  One BUY per book per day. Selling is unlimited. Nothing is forced.
  Long or flat only. No shorting, no leverage, no margin.
  Position size set by a 1% risk rule the model cannot override.

CONTROLS (do not delete)
  RANDOM - coin flip
  HODL   - buys once and NEVER sells; the "what if you did nothing" benchmark
  CROSS  - moving-average crossover, no AI at all
"""

import os, csv, json, time, random, statistics
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests

# ───────────────────────────── settings ─────────────────────────────

# ─── BUMP THIS whenever you change anything. ───
# Doing so archives the current failures to docs/fails.csv and zeroes the
# counter, so the dashboard shows failures caused by THIS version only.
VERSION = "2026-08-18b"

SYMBOL       = "ETH/EUR"
TIMEFRAME    = "1h"
CAPITAL      = 100.0
RISK_PCT     = 0.01
FEE_RATE     = 0.001
SLIPPAGE     = 0.0005
ATR_LEN      = 14
STOP_MULT    = 2.5           # raise toward 5-6 to cut fee drag and stop-outs
WINDOW       = 48
MIN_NOTIONAL = 5.0
PHASE_A_BARS = 720
PHASE_B_BARS = 720
HISTORY_MAX  = 800
CANDLE_MAX   = 500
TRADES_MAX   = 500
FAILLOG_MAX  = 2000          # pending failures held in state before archiving
ERR_CHARS    = 250           # how much of an error message to keep

PROVIDER    = "groq"
GROQ_M      = "openai/gpt-oss-20b"
GEMINI_M    = "gemini-2.5-flash"
ANTHROPIC_M = "claude-sonnet-4-6"
CALL_GAP    = 7.0

EXCHANGES = ["kraken", "coinbaseexchange", "bitstamp"]

STATE = Path("state.json")
DATA  = Path("docs/data.json")
FAILS = Path("docs/fails.csv")

STRATEGIES = [
    ("BASE",   ""),
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
CTRLS = ("RANDOM", "HODL", "CROSS")

# ───────────────────────────── state ────────────────────────────────

def blank(name, kind="ai"):
    return dict(name=name, kind=kind, cash=CAPITAL, pos=None, trades=[],
                peak=CAPITAL, last_buy="", phaseA_end=None, confs=[])

def load():
    if STATE.exists():
        s = json.loads(STATE.read_text())
        s.setdefault("candles", [])
        s.setdefault("fail_log", [])       # failures not yet archived
        s.setdefault("fails_all", s.get("fails", 0))
        s.setdefault("version", "")
        s.setdefault("archived", 0)
        return s
    books = {b: blank(b) for b in BOOKS}
    for c in CTRLS:
        books[c] = blank(c, kind="ctrl")
    return dict(bar=0, last_ts=None, phase="A", winner=None, done=False,
                calls=0, calls_today=0, day="", fails=0, fails_all=0,
                version="", archived=0, fail_log=[], books=books,
                history=[], candles=[], recent=[])

# ───────────────────── failure archiving ────────────────────────────

def archive_fails(s):
    """Append pending failures to docs/fails.csv, then clear the live log.
    Called only when VERSION changes. Nothing is destroyed - the CSV keeps
    everything, and fails_all still counts the lifetime total."""
    rows = s.get("fail_log", [])
    if rows:
        FAILS.parent.mkdir(parents=True, exist_ok=True)
        first = not FAILS.exists()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with FAILS.open("a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["archived_at", "from_version", "bar_time",
                            "book", "action", "detail"])
            for r in rows:
                w.writerow([stamp, r.get("v", s.get("version", "")), r.get("t", ""),
                            r.get("book", ""), r.get("act", ""), r.get("why", "")])
        s["archived"] = s.get("archived", 0) + len(rows)
    s["fail_log"] = []
    s["fails"] = 0
    s["recent"] = [r for r in s.get("recent", [])
                   if r.get("act") not in ("FAIL", "BLOCK")]
    return len(rows)

# ──────────────────────────── indicators ────────────────────────────

def atr(c, n=ATR_LEN):
    tr = [max(c[i][2]-c[i][3], abs(c[i][2]-c[i-1][4]), abs(c[i][3]-c[i-1][4]))
          for i in range(1, len(c))]
    return statistics.fmean(tr[-n:]) if tr else c[-1][4]*0.01

def sma(c, n):
    return statistics.fmean([x[4] for x in c[-n:]])

def eq(b, px):
    return b["cash"] + (b["pos"]["qty"]*px if b["pos"] else 0.0)

# ────────────────────────── paper execution ─────────────────────────

def buy(b, px, stop, day, ts, conf=None, why=""):
    if b["last_buy"] == day:
        return False                       # one buy per day. Hard rule.
    e = eq(b, px)
    stop = min(stop, px * 0.99)            # at least 1% away or fees eat it
    dist = px - stop
    if dist <= 0:
        return False
    afford = b["cash"] / (px * (1 + SLIPPAGE) * (1 + FEE_RATE) * 1.01)
    qty = min((e * RISK_PCT) / dist, afford)
    if qty * px < MIN_NOTIONAL:
        return False
    fill = px * (1 + SLIPPAGE)
    cost = qty * fill
    fee  = cost * FEE_RATE
    if cost + fee > b["cash"]:
        return False
    b["cash"] -= cost + fee
    b["pos"]   = dict(qty=qty, entry=fill, stop=stop, risk=qty*dist,
                      conf=conf, t=ts, why=why, eq_in=round(e, 2))
    b["last_buy"] = day
    return True

def sell(b, px, ts, why=""):
    p = b["pos"]
    if not p:
        return 0.0
    fill  = px * (1 - SLIPPAGE)
    gross = p["qty"] * fill
    fee   = gross * FEE_RATE
    b["cash"] += gross - fee
    entry_fee = p["qty"] * p["entry"] * FEE_RATE     # entry fee counts too
    pnl = (fill - p["entry"]) * p["qty"] - fee - entry_fee
    b["trades"].append(dict(
        tin=p.get("t", ""), tout=ts,
        px_in=round(p["entry"], 2), px_out=round(fill, 2),
        stop=round(p["stop"], 2), qty=round(p["qty"], 6),
        size=round(p["qty"] * p["entry"], 2),
        risk=round(p["risk"], 2),
        pnl=round(pnl, 2),
        R=round(pnl / p["risk"], 2) if p["risk"] else 0,
        eq_in=p.get("eq_in", CAPITAL), eq_out=round(b["cash"], 2),
        conf=p.get("conf"),
        why_in=p.get("why", "")[:60], why_out=why[:60]))
    if p.get("conf") is not None:
        b["confs"].append(dict(conf=p["conf"], won=pnl > 0))
    b["pos"] = None
    return pnl

# ─────────────────────────── model call ─────────────────────────────

def call_model(prompt):
    if PROVIDER == "groq":
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip(),
                     "content-type": "application/json"},
            json={"model": GROQ_M, "max_completion_tokens": 1500,
                  "temperature": 1.0,
                  "reasoning_effort": "low",      # reasoning tokens were eating the budget
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=45)
        if r.status_code != 200:
            if r.status_code == 429:
                time.sleep(10)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = r.json()["choices"][0]["message"]["content"]

    elif PROVIDER == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_M}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip(),
                     "content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 400, "temperature": 1.0,
                                       "responseMimeType": "application/json",
                                       "thinkingConfig": {"thinkingBudget": 0}}},
            timeout=45)
        if r.status_code != 200:
            if r.status_code == 429:
                time.sleep(10)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]

    else:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(),
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_M, "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=45)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:ERR_CHARS]}")
        txt = "".join(x.get("text", "") for x in r.json()["content"])

    txt = txt.replace("```json", "").replace("```", "").strip()
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])

def build_prompt(strat, b, candles, a, can_buy):
    closes = ",".join(f"{c[4]:.2f}" for c in candles[-WINDOW:])
    px = candles[-1][4]
    p  = b["pos"]
    if p:
        pos  = (f"HOLDING since {p['entry']:.2f}. Your stop is at {p['stop']:.2f}. "
                f"Price is currently {'above' if px > p['stop'] else 'BELOW'} that stop.")
        opts = '"hold" or "close"'
    else:
        pos  = "FLAT (no position)."
        opts = '"hold" or "buy"' if can_buy else '"hold" (no buy left today)'
    return f"""You trade spot {SYMBOL} in euros. You can only be long or flat. No shorting, no leverage.
{HINT[strat]}

Last {WINDOW} hourly closes, oldest first: {closes}
Current price: {px:.2f}
Typical hourly move (ATR{ATR_LEN}): {a:.2f}
Your position: {pos}
You may buy at most once per day.

Your reason must match the numbers above. Do not claim a stop was hit unless
the price is actually below it.

Position size is decided for you by a fixed risk rule. Do not choose size.

Reply with ONLY a json object in exactly this form, nothing else:
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
        # every entry carries the version that produced it, so a failure can
        # always be traced to a specific code state - even before archiving
        rec = dict(t=iso[5:], book=book, act=act, why=why, v=VERSION)
        s["recent"].insert(0, rec)
        if act in ("FAIL", "BLOCK"):
            s["fail_log"].append(rec)
            s["fail_log"] = s["fail_log"][-FAILLOG_MAX:]
            s["fails"] += 1
            s["fails_all"] = s.get("fails_all", 0) + 1

    # keep OHLC for the chart tab
    seen = {c[0] for c in s["candles"]}
    for c in candles:
        if c[0] not in seen:
            s["candles"].append([c[0], round(c[1], 2), round(c[2], 2),
                                 round(c[3], 2), round(c[4], 2)])
    s["candles"] = sorted(s["candles"], key=lambda c: c[0])[-CANDLE_MAX:]

    for b in books.values():
        if b["pos"] and b["name"] != "HODL" and low <= b["pos"]["stop"]:
            sp = b["pos"]["stop"]
            pnl = sell(b, sp, iso, "stop hit")
            note(b["name"], "STOP", f"stop {sp:.2f} hit ({pnl:+.2f} EUR)")

    h = books["HODL"]
    if not h["pos"] and h["cash"] > MIN_NOTIONAL:
        fill = px * (1 + SLIPPAGE)
        qty  = h["cash"] / (fill * (1 + FEE_RATE))
        h["cash"] -= qty * fill * (1 + FEE_RATE)
        h["pos"] = dict(qty=qty, entry=fill, stop=0.0, risk=1e9, t=iso,
                        why="bought once, never sells - the do-nothing benchmark",
                        eq_in=CAPITAL)

    cb = books["CROSS"]
    if len(candles) >= 30:
        fast, slow = sma(candles, 10), sma(candles, 30)
        if fast > slow and not cb["pos"]:
            if buy(cb, px, px - STOP_MULT * a, day, iso, why="10 crossed above 30"):
                note("CROSS", "BUY", "10 crossed above 30")
        elif fast < slow and cb["pos"]:
            pnl = sell(cb, px, iso, "crossed back below")
            note("CROSS", "CLOSE", f"crossed back ({pnl:+.2f} EUR)")

    active = [s["winner"]] if (s["phase"] == "B" and s["winner"]) else BOOKS
    entered = False
    for name in active:
        b = books[name]
        strat = name.rsplit("-", 1)[0]
        can_buy = b["last_buy"] != day

        if not b["pos"] and not can_buy:
            continue

        try:
            d = call_model(build_prompt(strat, b, candles, a, can_buy))
            s["calls"] += 1
            s["calls_today"] += 1
        except Exception as e:
            note(name, "FAIL", str(e)[:ERR_CHARS])
            time.sleep(CALL_GAP)
            continue
        time.sleep(CALL_GAP)

        act  = str(d.get("action", "hold")).lower().strip()
        why  = str(d.get("reason", ""))[:60]
        conf = d.get("confidence")
        conf = conf if isinstance(conf, (int, float)) else None

        if act == "buy" and not b["pos"]:
            if not can_buy:
                note(name, "BLOCK", "tried to buy twice in one day")
            elif buy(b, px, px - STOP_MULT * a, day, iso, conf, why):
                entered = True
                note(name, "BUY", why)
        elif act == "close" and b["pos"]:
            pnl = sell(b, px, iso, why)
            note(name, "CLOSE", f"{why} ({pnl:+.2f} EUR)")
        elif act not in ("buy", "hold", "close"):
            note(name, "FAIL", f"model returned an unknown action: '{act[:40]}'")

    rb = books["RANDOM"]
    if entered and not rb["pos"] and random.random() < 0.5:
        buy(rb, px, px - STOP_MULT * a, day, iso, why="coin flip: enter")
    elif rb["pos"] and random.random() < 0.04:
        sell(rb, px, iso, "coin flip: exit")

    for b in books.values():
        b["peak"] = max(b["peak"], eq(b, px))

    s["history"].append(dict(t=iso, px=round(px, 2),
                             eq={n: round(eq(b, px), 2) for n, b in books.items()}))
    s["history"] = s["history"][-HISTORY_MAX:]
    s["recent"]  = s["recent"][:40]
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
    rows, all_trades = [], []

    for n, b in books.items():
        tr = b["trades"]
        e  = eq(b, px)
        p  = b["pos"]
        rows.append(dict(
            name=n, kind=b["kind"],
            eur=round(e, 2), pnl=round(e - CAPITAL, 2),
            ret=round((e / CAPITAL - 1) * 100, 2), cash=round(b["cash"], 2),
            n=len(tr),
            win=round(100 * sum(1 for t in tr if t["pnl"] > 0) / len(tr)) if tr else 0,
            avgR=round(statistics.fmean([t["R"] for t in tr]), 2) if tr else 0,
            best=round(max((t["pnl"] for t in tr), default=0), 2),
            worst=round(min((t["pnl"] for t in tr), default=0), 2),
            dd=round((b["peak"] - e) / b["peak"] * 100, 1) if b["peak"] else 0,
            hint=HINT.get(n.rsplit("-", 1)[0], ""),
            open=dict(px_in=round(p["entry"], 2), t=p.get("t", ""),
                      why=p.get("why", ""), stop=round(p["stop"], 2),
                      size=round(p["qty"] * p["entry"], 2),
                      upl=round((px - p["entry"]) * p["qty"], 2),
                      uplpc=round((px - p["entry"]) / p["entry"] * 100, 2)) if p else None))
        for t in tr:
            all_trades.append(dict(book=n, **t))

    rows.sort(key=lambda x: -x["eur"])
    all_trades.sort(key=lambda t: t["tout"], reverse=True)
    all_trades = all_trades[:TRADES_MAX]

    gaps = []
    for name, _ in STRATEGIES:
        A, B = books.get(f"{name}-A"), books.get(f"{name}-B")
        if A and B:
            ea, eb = eq(A, px), eq(B, px)
            gaps.append(dict(s=name, a=round(ea, 2), b=round(eb, 2),
                             gap=round(abs(ea - eb), 2)))
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
        currency="EUR", version=VERSION,
        bar=s["bar"], phase=s["phase"], winner=s["winner"], done=s["done"],
        phaseA=PHASE_A_BARS, phaseB=PHASE_B_BARS,
        provider=PROVIDER,
        model=GROQ_M if PROVIDER == "groq" else (GEMINI_M if PROVIDER == "gemini" else ANTHROPIC_M),
        calls=s["calls"], today=s["calls_today"],
        fails=s["fails"], fails_all=s.get("fails_all", 0),
        archived=s.get("archived", 0),
        floor=floor, twins=gaps, books=rows, calib=calib, trades=all_trades,
        candles=s["candles"],
        history=[dict(t=h["t"][5:], px=h["px"], eq=h["eq"]) for h in s["history"]],
        recent=s["recent"][:30]), separators=(",", ":")))

# ─────────────────────────── prices ─────────────────────────────────

def fetch_candles():
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

    # version bump -> archive failures, reset the live counter
    if s.get("version") != VERSION:
        pending = len(s.get("fail_log", []))
        archive_fails(s)
        old = s.get("version") or "(none)"
        s["version"] = VERSION
        print(f"version {old} -> {VERSION}: archived {pending} failures to {FAILS}, "
              f"counter reset (lifetime total {s.get('fails_all', 0)})")

    if s.get("done"):
        print("experiment complete")
        emit_px = s["history"][-1]["px"] if s["history"] else CAPITAL
        emit(s, emit_px)
        STATE.write_text(json.dumps(s, separators=(",", ":")))
        return

    c = fetch_candles()
    if c[-1][0] == s["last_ts"]:
        print("no new candle, refreshing dashboard only")
        emit(s, c[-1][4])
        STATE.write_text(json.dumps(s, separators=(",", ":")))
        return

    s["last_ts"] = c[-1][0]
    step(s, c)
    STATE.write_text(json.dumps(s, separators=(",", ":")))
    emit(s, c[-1][4])
    print(f"bar {s['bar']} phase {s['phase']} px {c[-1][4]:.2f} EUR "
          f"calls {s['calls_today']} today, fails {s['fails']} this version "
          f"({s.get('fails_all', 0)} lifetime)")

if __name__ == "__main__":
    main()
