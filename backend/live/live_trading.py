# ===============================================================
#  ORBOFI AGENTVERSE — LIVE PERPS TRADING ENGINE
#  Personality-Driven Multi-Agent Perps Trading (REAL ORDERS)
#
#  - Celebrity/personality prompt injection per agent
#  - Shared on-chain wallet across all agents
#  - Slippage on executions
#  - Funding payments (simulated)
#  - Maintenance margin & liquidation (simulated risk layer)
#  - TP / SL based on margin
#  - Per-trade margin caps
#  - Writes arena_state.json for the Flask dashboard
# ===============================================================

import os
import json
import math
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from asyncio import Semaphore, Lock
from typing import Dict, Any, Optional

import requests
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from eth_abi import encode
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

from openai import OpenAI

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_BASE = "https://fapi.asterdex.com/fapi/v1"
ASTERDEX_HOST = "https://fapi.asterdex.com"
ORDER_URL = "/fapi/v3/order"

# Symbols you want the agents to trade live
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
]

INTERVAL = "1h"              # indicator timeframe (matches backtest)
STATE_FILE = os.getenv("ARENA_STATE_FILE", "arena_state.json")
AGENTS_FILE = os.getenv("AGENTS_FILE", "agents.json")

# Live loop config
LIVE_SLEEP_SECONDS = 60       # seconds between decision cycles (you can tune this)
OHLCV_LIMIT = 200             # candles per symbol for indicators

# Risk / capital config (same semantics as backtest)
QUOTE_BALANCE_START = 50.0

MAX_ORDER_USD = 25.0          # absolute per-trade margin cap
MAX_LEVERAGE = 20
MAX_TRADE_PCT = 0.25          # % of equity allowed as margin per trade
MAX_CONCURRENT_AGENTS = 10
HISTORY_LIMIT = 400
TAKER_FEE = 0.0005

# TP/SL based on % of margin
TAKE_PROFIT_PCT = 15.0        # +15% on margin
STOP_LOSS_PCT = -25.0         # -25% on margin

# Realism / risk parameters
MAINTENANCE_MARGIN_RATIO = 0.5     # if position equity < 50% of margin → liquidation
FUNDING_INTERVAL_STEPS = 8         # apply funding every N live iterations (approx. N hours if INTERVAL is 1h)
BASE_FUNDING_RATE = 0.0001         # 0.01% per funding event (approx)
SLIPPAGE_BASIS_POINTS = 2          # 2 bps = 0.02% slippage

# AsterDex shared wallet (all agents share this)
ASTER_USER = os.getenv("ASTER_USER")
ASTER_SIGNER = os.getenv("ASTER_SIGNER")
ASTER_PRIVKEY = os.getenv("ASTER_PRIVKEY")

if not (ASTER_USER and ASTER_SIGNER and ASTER_PRIVKEY):
    raise RuntimeError("ASTER_USER, ASTER_SIGNER, and ASTER_PRIVKEY env vars are required for live trading.")

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for live trading.")

client = OpenAI(api_key=OPENAI_KEY)

# Global lock to protect shared in-memory state in async loop
state_lock: Lock = Lock()


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def format_dt(dt: datetime) -> str:
    """Format datetime as UTC ISO8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def atomic_save_json(path: str, data: Any) -> None:
    """Write JSON atomically to avoid partial writes."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def get_ohlcv_live(symbol: str, interval: str = INTERVAL, limit: int = OHLCV_LIMIT) -> pd.DataFrame:
    """Fetch recent OHLCV data from AsterDex /klines for live indicators."""
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{API_BASE}/klines", params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(
        r.json(),
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tb_base",
            "tb_quote",
            "ignore",
        ],
    )
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Compute TA indicators for the most recent candle in a DataFrame (same as backtest)."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema20 = EMAIndicator(close, 20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close, 50).ema_indicator().iloc[-1]
    macd_obj = MACD(close)
    macd_hist = macd_obj.macd_diff().iloc[-1]
    rsi = RSIIndicator(close, 14).rsi().iloc[-1]
    atr = AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
    bb = BollingerBands(close, 20, 2)
    bb_width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb.bollinger_mavg().iloc[-1]

    if ema20 > ema50:
        trend = "BULLISH"
    elif ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "price": float(close.iloc[-1]),
        "rsi": float(rsi),
        "macd_hist": float(macd_hist),
        "atr": float(atr),
        "bb_width": float(bb_width),
        "trend": trend,
    }


def summarize_market_for_llm(market: Dict[str, Dict[str, float]]) -> str:
    """Convert TA snapshot into a compact text summary for the LLM."""
    lines = []
    for symbol, d in market.items():
        lines.append(
            f"{symbol}: {d['trend']}. RSI {d['rsi']:.1f}, MACD {d['macd_hist']:.2f}, "
            f"ATR {d['atr']:.2f}, BB width {(d['bb_width'] * 100):.1f}%."
        )
    return "\n".join(lines)


def apply_slippage(price: Optional[float], side: str, action: str) -> Optional[float]:
    """
    Apply simple bid/ask spread slippage:
    - LONGs pay slightly more when opening, receive slightly less when closing.
    - SHORTs open slightly below and close slightly above.
    """
    if price is None or price <= 0:
        return price

    slip = SLIPPAGE_BASIS_POINTS / 10_000.0  # bps → fraction
    side = side.upper()
    action = action.upper()

    if action == "OPEN":
        if side == "LONG":
            return price * (1 + slip)
        if side == "SHORT":
            return price * (1 - slip)
    elif action == "CLOSE":
        if side == "LONG":
            return price * (1 - slip)
        if side == "SHORT":
            return price * (1 + slip)

    return price


def estimate_funding_rates(market: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Simple deterministic funding model:
    - If trend is BULLISH → positive funding (longs pay shorts)
    - If trend is BEARISH → negative funding (shorts pay longs)
    - If trend is NEUTRAL → 0
    """
    rates: Dict[str, float] = {}
    for sym, d in market.items():
        trend = d.get("trend", "NEUTRAL")
        if trend == "BULLISH":
            sign = 1.0
        elif trend == "BEARISH":
            sign = -1.0
        else:
            sign = 0.0
        rates[sym] = BASE_FUNDING_RATE * sign
    return rates


# --------------------------------------------------
# ASTERDEX SIGNING & ORDER SENDING
# --------------------------------------------------

def _trim_dict(my_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in list(my_dict.items()):
        if isinstance(value, list):
            new_value = []
            for item in value:
                if isinstance(item, dict):
                    new_value.append(json.dumps(_trim_dict(item)))
                else:
                    new_value.append(str(item))
            my_dict[key] = json.dumps(new_value)
        elif isinstance(value, dict):
            my_dict[key] = json.dumps(_trim_dict(value))
        else:
            my_dict[key] = str(value)
    return my_dict


def sign_payload(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build signed payload for /fapi/v3/order:
    - Add recvWindow, timestamp, nonce
    - Encode (json_str, user, signer, nonce)
    - keccak
    - sign hash with private key
    """
    nonce = math.trunc(time.time() * 1_000_000)

    # remove None values
    params = {k: v for k, v in params.items() if v is not None}
    params["recvWindow"] = 50000
    params["timestamp"] = int(round(time.time() * 1000))

    _trim_dict(params)
    json_str = json.dumps(params, sort_keys=True).replace(" ", "").replace("'", "\"")

    encoded = encode(["string", "address", "address", "uint256"], [json_str, ASTER_USER, ASTER_SIGNER, nonce])
    keccak_hex = Web3.keccak(encoded).hex()

    signable_msg = encode_defunct(hexstr=keccak_hex)
    signed = Account.sign_message(signable_msg, private_key=ASTER_PRIVKEY)

    params["nonce"] = nonce
    params["user"] = ASTER_USER
    params["signer"] = ASTER_SIGNER
    params["signature"] = "0x" + signed.signature.hex()

    return params


def send_order(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a REAL order to AsterDex using signed payload.
    """
    url = ASTERDEX_HOST + ORDER_URL
    payload = sign_payload(params)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "OrbofiAgentverse/1.0",
    }
    res = requests.post(url, data=payload, headers=headers)
    try:
        j = res.json()
    except Exception:
        j = {"raw": res.text}

    print(
        f"🚀 [{datetime.utcnow().isoformat()}] "
        f"Order {params.get('side')} {params.get('symbol')} qty={params.get('quantity')} "
        f"type={params.get('type')} price={params.get('price')} → {j}"
    )
    return j


def build_order_params(symbol: str, side: str, action: str, qty: float, price: float) -> Dict[str, Any]:
    """
    Map LONG/SHORT + OPEN/CLOSE into AsterDex order params.
    - LONG + OPEN   → BUY,  reduceOnly=False
    - LONG + CLOSE  → SELL, reduceOnly=True
    - SHORT + OPEN  → SELL, reduceOnly=False
    - SHORT + CLOSE → BUY,  reduceOnly=True
    """
    side = side.upper()
    action = action.upper()

    if action == "OPEN":
        if side == "LONG":
            order_side = "BUY"
        else:  # SHORT
            order_side = "SELL"
        reduce_only = False
    else:  # CLOSE
        if side == "LONG":
            order_side = "SELL"
        else:
            order_side = "BUY"
        reduce_only = True

    # NOTE: For simplicity, we use LIMIT orders at the current exec price.
    # You can adjust to MARKET if supported: type="MARKET" and omit price/timeInForce.
    return {
        "symbol": symbol,
        "positionSide": "BOTH",
        "type": "LIMIT",
        "side": order_side,
        "timeInForce": "GTC",
        "quantity": str(qty),
        "price": price,
        "reduceOnly": reduce_only,
    }


# --------------------------------------------------
# PORTFOLIO + POSITION (SAME AS BACKTEST, PLUS LIVE ORDERS)
# --------------------------------------------------

@dataclass
class PerpPosition:
    side: str
    leverage: float
    entry: float
    qty: float
    margin: float


class AgentPortfolio:
    """Represents a single agent's portfolio and trading actions."""

    def __init__(self, name: str, aid: str, prompt: str, img: Optional[str] = None) -> None:
        self.name = name
        self.id = aid
        self.prompt = prompt
        self.img = img

        self.cash: float = QUOTE_BALANCE_START
        self.positions: Dict[str, PerpPosition] = {}
        self.trade_log: list[Dict[str, Any]] = []
        self.total_value: float = self.cash
        self.pnl: float = 0.0
        self.last_update: Optional[str] = None
        self.last_decision: Optional[Dict[str, Any]] = None

        # Extra metrics for realism
        self.funding_pnl: float = 0.0
        self.fees_paid: float = 0.0

    # ------------ Core math ------------

    def _fee(self, notional: float) -> float:
        fee = abs(notional) * TAKER_FEE
        self.fees_paid += fee
        return fee

    def _qty_from_margin(self, margin: float, lev: float, price: float) -> float:
        return (margin * lev) / max(price, 1e-12)

    def value(self, prices: Dict[str, float]) -> float:
        """Recompute equity using mark prices for all open positions."""
        eq = self.cash
        for sym, p in self.positions.items():
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue
            upnl = (px - p.entry) * p.qty if p.side == "LONG" else (p.entry - px) * p.qty
            eq += p.margin + upnl

        self.total_value = float(eq)
        self.pnl = float(eq - QUOTE_BALANCE_START)
        return self.total_value

    # ------------ Logging ------------

    def _log_trade(self, t: Dict[str, Any]) -> Dict[str, Any]:
        t["agent"] = self.name
        t["agent_img"] = self.img
        self.trade_log.append(t)
        return t

    # ------------ Trade actions (OPEN / CLOSE / HOLD) ------------
    # These are identical in logic to the backtest, but now they
    # also send REAL orders to AsterDex via send_order().

    def open(
        self,
        sym: str,
        side: str,
        lev: float,
        margin: float,
        price: float,
        reason: str,
        reasoning: str,
        thinking: Any,
        t: datetime,
        user_prompt: str = "",
        exit_reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        side = side.upper()
        lev = float(lev)
        lev = min(max(lev, 1.0), MAX_LEVERAGE)

        margin = float(margin)

        # Enforce per-trade margin caps (absolute and % of equity)
        equity = self.total_value if self.total_value > 0 else QUOTE_BALANCE_START
        max_margin_equity = equity * MAX_TRADE_PCT
        max_margin_abs = MAX_ORDER_USD
        effective_max = max(0.0, min(max_margin_equity, max_margin_abs))

        if effective_max <= 0:
            return None

        if margin > effective_max:
            margin = effective_max

        if margin <= 0 or price <= 0:
            return None

        notional = margin * lev
        fee = self._fee(notional)

        if self.cash < margin + fee:
            return None

        qty = self._qty_from_margin(margin, lev, price)
        cur = self.positions.get(sym)

        # Flip side if needed
        if cur and cur.side != side:
            self.close(
                sym,
                price,
                cur.margin,
                "Flipped side",
                reasoning,
                thinking,
                t,
                exit_reason="Closed opposing side",
                user_prompt=user_prompt,
            )
            cur = None

        # Deduct cash now that trade is valid (local accounting)
        self.cash -= (margin + fee)

        # Update local position
        if cur and cur.side == side:
            total_qty = cur.qty + qty
            cur.entry = (cur.entry * cur.qty + price * qty) / total_qty
            cur.qty = total_qty
            cur.margin += margin
        else:
            self.positions[sym] = PerpPosition(side, lev, price, qty, margin)

        # Send REAL order
        order_params = build_order_params(sym, side, "OPEN", qty, price)
        order_resp = send_order(order_params)

        return self._log_trade(
            {
                "timestamp": format_dt(t),
                "type": "OPEN",
                "symbol": sym,
                "price": price,
                "side": side,
                "leverage": lev,
                "amount_margin_usd": margin,
                "qty": qty,
                "reason": reason,
                "exit_strategy_reason": exit_reason,
                "user_prompt": user_prompt,
                "exchange_order": order_resp,
            }
        )

    def close(
        self,
        sym: str,
        price: float,
        margin_close: Optional[float],
        reason: str,
        reasoning: str,
        thinking: Any,
        t: datetime,
        exit_reason: str = "",
        user_prompt: str = "",
    ) -> Optional[Dict[str, Any]]:
        if sym not in self.positions or price <= 0:
            return None

        p = self.positions[sym]
        margin_close = float(margin_close) if margin_close else 0.0
        frac = 1.0 if not margin_close else min(1.0, margin_close / p.margin)

        qty_close = p.qty * frac
        margin_release = p.margin * frac
        fee = self._fee(qty_close * price)
        realized = (
            (price - p.entry) * qty_close
            if p.side == "LONG"
            else (p.entry - price) * qty_close
        )

        self.cash += margin_release + realized - fee

        p.qty -= qty_close
        p.margin -= margin_release
        if p.qty <= 1e-8:
            del self.positions[sym]

        # Send REAL order
        order_params = build_order_params(sym, p.side, "CLOSE", qty_close, price)
        order_resp = send_order(order_params)

        return self._log_trade(
            {
                "timestamp": format_dt(t),
                "type": "CLOSE",
                "symbol": sym,
                "price": price,
                "side": p.side,
                "leverage": p.leverage,
                "qty": qty_close,
                "realized_pnl": realized,
                "reason": reason,
                "exit_strategy_reason": exit_reason,
                "user_prompt": user_prompt,
                "exchange_order": order_resp,
            }
        )

    def hold(
        self,
        sym: str,
        reason: str,
        reasoning: str,
        thinking: Any,
        t: datetime,
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        # No real order; just log HOLD action for analytics
        return self._log_trade(
            {
                "timestamp": format_dt(t),
                "type": "HOLD",
                "symbol": sym,
                "reason": reason,
                "user_prompt": user_prompt,
            }
        )

    # ------------ Risk / controls ------------

    def serializable_positions(self) -> Dict[str, Dict[str, Any]]:
        return {s: vars(p) for s, p in self.positions.items()}

    def check_liquidations(self, prices: Dict[str, float], t: datetime) -> None:
        """
        If position equity (margin + PnL) drops below
        MAINTENANCE_MARGIN_RATIO * margin, force full liquidation.
        """
        for sym, p in list(self.positions.items()):
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue

            upnl = (px - p.entry) * p.qty if p.side == "LONG" else (p.entry - px) * p.qty
            position_equity = p.margin + upnl

            if position_equity <= p.margin * MAINTENANCE_MARGIN_RATIO:
                self.close(
                    sym,
                    px,
                    p.margin,
                    "Liquidation: margin below maintenance",
                    "",
                    "",
                    t,
                    exit_reason="Auto liquidation",
                    user_prompt="[AUTO-LIQ]",
                )

    def apply_funding(
        self,
        prices: Dict[str, float],
        funding_rates: Optional[Dict[str, float]],
        t: datetime,
    ) -> None:
        """
        Apply funding payments to all open positions.
        Convention: positive rate → LONGs pay SHORTs.
        Adjusts cash & funding_pnl, does not alter trade_log.
        """
        if not funding_rates:
            return

        total_funding = 0.0
        for sym, p in self.positions.items():
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue

            rate = funding_rates.get(sym, 0.0)
            if rate == 0.0:
                continue

            notional = p.qty * px
            if p.side == "LONG":
                funding = -notional * rate  # longs pay when rate > 0
            else:
                funding = notional * rate   # shorts receive when rate > 0

            self.cash += funding
            total_funding += funding

        if abs(total_funding) > 0:
            self.funding_pnl += total_funding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "img": self.img,
            "cash": self.cash,
            "positions": self.serializable_positions(),
            "pnl": self.pnl,
            "total_value": self.total_value,
            "trade_log": self.trade_log[-30:],
            "last_update": self.last_update,
            "last_decision": self.last_decision,
            # Extra analytics
            "funding_pnl": self.funding_pnl,
            "fees_paid": self.fees_paid,
        }


# --------------------------------------------------
# PERSONALITY PROMPT BUILDER (SAME AS SIMULATION)
# --------------------------------------------------

def build_persona_prompt(portfolio: AgentPortfolio) -> str:
    return f"""
You are {portfolio.name}, a crypto perpetuals trader with the following personality:
{portfolio.prompt}

RISK & EXECUTION CONSTRAINTS (CRITICAL):
- You trade linear USDT-margined perpetual futures on major crypto pairs.
- You may not use more than ${MAX_ORDER_USD:.2f} as margin in a single trade.
- You may not use more than {MAX_TRADE_PCT * 100:.1f}% of your total equity as margin per trade.
- Your leverage must always be between 1x and {MAX_LEVERAGE}x.
- There are trading fees and slippage, so over-trading is dangerous.

REASONING STYLE:
- Always reason as this personality.
- When giving your reason, explicitly reference:
  - The current UTC time of the decision.
  - Your current holdings (e.g. what you're already long/short).
- Example phrasing:
  "At 2025-11-03T09:00Z, with an existing LONG on BTCUSDT and no ETHUSDT exposure, I decide to..."
  "At this hour, with no open positions and $50 equity, I prefer to stay in cash because..."

OUTPUT FORMAT:
Respond strictly in JSON:
{{
 "action": "OPEN|CLOSE|HOLD",
 "symbol": "<symbol>",
 "side": "LONG|SHORT",
 "leverage": <1-20>,
 "quoteOrderQty": <margin_usd>,
 "reason": "<concise, in-character reason mentioning time and holdings>",
 "exit_strategy_reason": "<exit logic or HOLD rationale>"
}}

Be concise, bold, and confident. Never break character. Never output anything that is not valid JSON.
"""


# --------------------------------------------------
# LLM DECISION (SAME AS SIMULATION)
# --------------------------------------------------

def llm_decision(sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Synchronous LLM decision call."""
    r = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


async def llm_decision_async(sys_prompt: str, user_prompt: str) -> tuple[Dict[str, Any], str, str]:
    """Async wrapper around llm_decision, with fallback HOLD on error."""
    try:
        res = await asyncio.to_thread(llm_decision, sys_prompt, user_prompt)
        return res, "", ""
    except Exception as e:  # noqa: BLE001
        print("⚠️ LLM failure:", e)
        return (
            {
                "action": "HOLD",
                "symbol": "",
                "side": "",
                "leverage": 1,
                "quoteOrderQty": 0,
                "reason": "Fallback HOLD due to LLM error.",
                "exit_strategy_reason": "",
            },
            "",
            "",
        )


# --------------------------------------------------
# LIVE MULTI-AGENT LOOP (SIMULATION LOGIC + REAL ORDERS)
# --------------------------------------------------

async def run_live() -> None:
    """
    Multi-agent live loop:
    - Fetches live OHLCV from AsterDex
    - Computes indicators (same as backtest)
    - Applies TP/SL, liquidation, funding
    - Asks LLM for each agent's decision
    - Sends REAL orders on OPEN/CLOSE via AsterDex API
    - Writes arena_state.json for visualization
    """

    print("🏁 Starting ORBOFI AGENTVERSE — LIVE PERPS TRADING")

    # Load agents
    if not os.path.exists(AGENTS_FILE):
        raise FileNotFoundError(f"Agents config file not found: {AGENTS_FILE}")

    with open(AGENTS_FILE, "r") as f:
        agents_cfg = json.load(f)

    agent_states: Dict[str, AgentPortfolio] = {
        a["agent_name"]: AgentPortfolio(
            a["agent_name"],
            a["agent_id"],
            a.get("agent_personality_and_data_prompt_injection", ""),
            a.get("img"),
        )
        for a in agents_cfg
    }

    trade_history: list[Dict[str, Any]] = []
    history: list[Dict[str, Any]] = []
    sem = Semaphore(MAX_CONCURRENT_AGENTS)

    step = 0

    while True:
        step += 1
        now = datetime.utcnow().replace(tzinfo=timezone.utc)

        try:
            # 1) Build live market snapshot for all symbols
            market: Dict[str, Dict[str, float]] = {}
            for symbol in SYMBOLS:
                df = get_ohlcv_live(symbol)
                if len(df) > 50:
                    market[symbol] = compute_indicators(df)

            prices: Dict[str, float] = {s: d["price"] for s, d in market.items()}
            if not prices:
                print("⚠️ No valid prices this cycle; skipping.")
                await asyncio.sleep(LIVE_SLEEP_SECONDS)
                continue

            market_text = summarize_market_for_llm(market)

            # 2) Funding rates periodically (simulated like in backtest)
            funding_rates = (
                estimate_funding_rates(market)
                if step > 0 and step % FUNDING_INTERVAL_STEPS == 0
                else None
            )

            async def run_agent(name: str, p: AgentPortfolio) -> None:
                async with sem:
                    # --- Auto TP/SL based on PnL vs margin ---
                    for sym, pos in list(p.positions.items()):
                        cur_px = prices.get(sym)
                        if not cur_px:
                            continue

                        upnl = (
                            (cur_px - pos.entry) * pos.qty
                            if pos.side == "LONG"
                            else (pos.entry - cur_px) * pos.qty
                        )
                        pct = (upnl / pos.margin) * 100 if pos.margin else 0.0

                        if pct >= TAKE_PROFIT_PCT:
                            exec_px = apply_slippage(cur_px, pos.side, "CLOSE")
                            p.close(
                                sym,
                                exec_px,
                                pos.margin,
                                f"TP +{pct:.2f}%",
                                "",
                                "",
                                now,
                                exit_reason=f"Auto TP +{pct:.2f}%",
                                user_prompt="[AUTO TP]",
                            )
                        elif pct <= STOP_LOSS_PCT:
                            exec_px = apply_slippage(cur_px, pos.side, "CLOSE")
                            p.close(
                                sym,
                                exec_px,
                                pos.margin,
                                f"SL {pct:.2f}%",
                                "",
                                "",
                                now,
                                exit_reason=f"Auto SL {pct:.2f}%",
                                user_prompt="[AUTO SL]",
                            )

                    # --- Liquidation checks (maintenance margin) ---
                    p.check_liquidations(prices, now)

                    # --- Apply funding (if due) ---
                    if funding_rates:
                        p.apply_funding(prices, funding_rates, now)

                    # Re-value after auto adjustments
                    p.value(prices)

                    # --- Persona-driven decision ---
                    sys_prompt = build_persona_prompt(p)
                    current_positions = json.dumps(p.serializable_positions(), indent=2)

                    user_prompt = (
                        f"Time (UTC): {format_dt(now)}\n"
                        f"Cash: ${p.cash:.2f}\n"
                        f"Equity: ${p.total_value:.2f}\n"
                        f"Open positions: {current_positions if current_positions != '{}' else 'None'}\n\n"
                        f"Market snapshot:\n{market_text}\n\n"
                        "When you decide, explicitly reference this time and your current holdings "
                        "in your reason to effectively complete your feedback loop."
                    )

                    decision, _, _ = await llm_decision_async(sys_prompt, user_prompt)

                    act = decision.get("action", "HOLD").upper()
                    sym = decision.get("symbol", "").upper()
                    side = decision.get("side", "").upper()
                    lev = float(decision.get("leverage", 1))
                    margin = float(decision.get("quoteOrderQty", 0))
                    raw_price = prices.get(sym)
                    reason = decision.get("reason", "")
                    exit_reason = decision.get("exit_strategy_reason", "")

                    async with state_lock:
                        if raw_price is None:
                            p.hold(sym, "No price data", "", "", now, user_prompt=user_prompt)

                        elif act == "OPEN":
                            exec_price = apply_slippage(raw_price, side, "OPEN")
                            trade = p.open(
                                sym,
                                side,
                                lev,
                                margin,
                                exec_price,
                                reason,
                                "",
                                "",
                                now,
                                user_prompt=user_prompt,
                                exit_reason=exit_reason,
                            )
                            p.value(prices)
                            p.last_decision = {
                                "action": act,
                                "symbol": sym,
                                "side": side,
                                "leverage": lev,
                                "margin_usd": margin,
                                "reason": reason,
                                "exit_strategy_reason": exit_reason,
                                "user_prompt": user_prompt,
                                "agent_img": p.img,
                            }
                            p.last_update = format_dt(now)
                            if trade:
                                trade.update({"pnl": p.pnl, "equity": p.total_value})
                                trade_history.append(trade)

                        elif act == "CLOSE":
                            pos = p.positions.get(sym)
                            position_side = pos.side if pos else side
                            exec_price = apply_slippage(raw_price, position_side, "CLOSE")

                            trade = p.close(
                                sym,
                                exec_price,
                                margin,
                                reason,
                                "",
                                "",
                                now,
                                exit_reason=exit_reason,
                                user_prompt=user_prompt,
                            )
                            p.value(prices)
                            p.last_decision = {
                                "action": act,
                                "symbol": sym,
                                "side": position_side,
                                "leverage": lev,
                                "margin_usd": margin,
                                "reason": reason,
                                "exit_strategy_reason": exit_reason,
                                "user_prompt": user_prompt,
                                "agent_img": p.img,
                            }
                            p.last_update = format_dt(now)
                            if trade:
                                trade.update({"pnl": p.pnl, "equity": p.total_value})
                                trade_history.append(trade)

                        else:
                            p.hold(sym, reason or "HOLD", "", "", now, user_prompt=user_prompt)
                            p.last_update = format_dt(now)

            # Run all agents concurrently for this timestep
            await asyncio.gather(*(run_agent(n, p) for n, p in agent_states.items()))

            # Snapshot arena state for visualization (same schema as backtest)
            history.append(
                {
                    "timestamp": format_dt(now),
                    "agents": {
                        n: {"pnl": p.pnl, "total_value": p.total_value}
                        for n, p in agent_states.items()
                    },
                }
            )

            out = {
                "schema_version": "1.0",
                "updated_at": format_dt(now),
                "trading_mode": "live",
                "interval": INTERVAL,
                "max_order_usd": MAX_ORDER_USD,
                "agents": {n: p.to_dict() for n, p in agent_states.items()},
                "history": history[-HISTORY_LIMIT:],
                "trade_history": trade_history[-2000:],
            }
            atomic_save_json(STATE_FILE, out)
            print(f"💾 Live step {step} | Trades logged={len(trade_history)}")

        except Exception as e:
            print("⚠️ Live loop error:", e)

        await asyncio.sleep(LIVE_SLEEP_SECONDS)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_live())
