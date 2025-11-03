# ===============================================================
#  ORBOFI AI - AGENTVERSE PERPS BACKTEST
#  Personality-Driven Multi-Agent Perps Simulation (TP/SL, Funding)
#
#  Features:
#   - Personality prompt injection per agent
#   - agent_img in trade_log, trade_history, last_decision
#   - Slippage on executions
#   - Funding payments
#   - Maintenance margin & liquidation
#   - Per-trade margin caps
#   - Async multi-agent backtest
# ===============================================================

import os
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from asyncio import Semaphore, Lock
from typing import Dict, Any, Optional

import pandas as pd
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from openai import OpenAI

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_BASE = "https://fapi.asterdex.com/fapi/v1"
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "ASTERUSDT",
]

INTERVAL = "1h"
DAYS_BACK = 10

STATE_FILE = os.getenv("ARENA_STATE_FILE", "arena_state.json")
AGENTS_FILE = os.getenv("AGENTS_FILE", "agents.json")

QUOTE_BALANCE_START = 50.0

# Risk caps
MAX_ORDER_USD = 25.0        # absolute per-trade margin cap
MAX_LEVERAGE = 20
MAX_TRADE_PCT = 0.25        # % of equity allowed as margin per trade

MAX_CONCURRENT_AGENTS = 10
HISTORY_LIMIT = 400

TAKER_FEE = 0.0005

# TP/SL based on % of margin
TAKE_PROFIT_PCT = 15.0      # +15% on margin
STOP_LOSS_PCT = -25.0       # -25% on margin

# Realism / risk parameters
MAINTENANCE_MARGIN_RATIO = 0.5   # if position equity < 50% of margin → liquidation
FUNDING_INTERVAL_HOURS = 8       # apply funding every N time steps
BASE_FUNDING_RATE = 0.0001       # 0.01% per funding event (approx)
SLIPPAGE_BASIS_POINTS = 2        # 2 bps = 0.02% slippage

# OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for backtest.")

client = OpenAI(api_key=OPENAI_KEY)

# Global lock to protect shared state in async loop
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


def get_ohlcv(
    symbol: str,
    interval: str = "1h",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV data from AsterDex /klines endpoint."""
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start:
        params["startTime"] = int(start.timestamp() * 1000)
    if end:
        params["endTime"] = int(end.timestamp() * 1000)

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
    """Compute TA indicators for the most recent candle in a DataFrame."""
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
# PORTFOLIO + POSITION
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

    # ------------ Trade actions ------------

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

        # Deduct cash now that trade is valid
        self.cash -= (margin + fee)

        if cur and cur.side == side:
            total_qty = cur.qty + qty
            cur.entry = (cur.entry * cur.qty + price * qty) / total_qty
            cur.qty = total_qty
            cur.margin += margin
        else:
            self.positions[sym] = PerpPosition(side, lev, price, qty, margin)

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
# PERSONALITY PROMPT BUILDER
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
# LLM DECISION
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

    

def letta_decision(agent_id, sys_prompt, user_prompt, retries=1):
    full_response = ""
    for attempt in range(retries + 1):
        try:
            print(f"\n{Fore.CYAN}🤖 Streaming Letta Decision for {DEFAULT_LETTA_AGENT_ID}{Style.RESET_ALL}")
            stream = client.agents.messages.create_stream(
                agent_id=DEFAULT_LETTA_AGENT_ID,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream_tokens=True,
            )

            for chunk in stream:
                print(chunk)
                if getattr(chunk, "message_type", None) == "assistant_message":
                    content_piece = getattr(chunk, "content", "")
                    full_response += content_piece
                    print(f"{Fore.YELLOW}{content_piece}{Style.RESET_ALL}", end="", flush=True)

                if getattr(chunk, "message_type", None) == 'reasoning_message':
                    reasoning_content_piece = getattr(chunk, "content", "")
                    full_response += reasoning_content_piece
                    print(f"{Fore.RED}{reasoning_content_piece}{Style.RESET_ALL}", end="", flush=True)


            print(f"\n{Fore.GREEN}✅ Stream complete.{Style.RESET_ALL}")
            cleaned = (
                full_response.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed = json.loads(cleaned)

            # ✅ Extract reasoning from the JSON itself
            reasoning_text = parsed.get("reason", "(no reason provided)")
            return cleaned, parsed, reasoning_text

        except Exception as e:
            print(f"{Fore.RED}⚠️ Letta failed (attempt {attempt+1}/{retries+1}): {e}{Style.RESET_ALL}")
            time.sleep(2)
    return {}, "(Letta failed — no reasoning)"


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
# BACKTEST CORE
# --------------------------------------------------

async def run_backtest() -> None:
    """Run the full multi-agent backtest and write arena_state.json snapshots."""
    print(f"\n🏁 Starting Personality-Driven Perps Backtest ({DAYS_BACK} days)")
    end = datetime.utcnow().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=DAYS_BACK)

    # Load data
    ohlcv_all: Dict[str, pd.DataFrame] = {
        s: get_ohlcv(s, INTERVAL, start, end) for s in SYMBOLS
    }

    if "BTCUSDT" not in ohlcv_all:
        raise RuntimeError("BTCUSDT data missing; cannot construct time axis for backtest.")

    timestamps = sorted(set(ohlcv_all["BTCUSDT"]["open_time"].dt.floor("h")))

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

    # Main time loop
    for step, t in enumerate(timestamps[50:]):
        # Build market snapshot up to current time
        market: Dict[str, Dict[str, float]] = {}
        for symbol, df in ohlcv_all.items():
            df_cut = df[df["open_time"] <= t]
            if len(df_cut) > 50:
                market[symbol] = compute_indicators(df_cut)

        prices: Dict[str, float] = {s: d["price"] for s, d in market.items()}
        if not prices:
            continue

        market_text = summarize_market_for_llm(market)

        # Funding rates every FUNDING_INTERVAL_HOURS steps
        funding_rates = (
            estimate_funding_rates(market)
            if step > 0 and step % FUNDING_INTERVAL_HOURS == 0
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
                            t,
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
                            t,
                            exit_reason=f"Auto SL {pct:.2f}%",
                            user_prompt="[AUTO SL]",
                        )

                # --- Liquidation checks (maintenance margin) ---
                p.check_liquidations(prices, t)

                # --- Apply funding (if due) ---
                if funding_rates:
                    p.apply_funding(prices, funding_rates, t)

                # Re-value after auto adjustments
                p.value(prices)

                # --- Persona-driven decision ---
                sys_prompt = build_persona_prompt(p)
                current_positions = json.dumps(p.serializable_positions(), indent=2)

                user_prompt = (
                    f"Time (UTC): {format_dt(t)}\n"
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
                        trade = p.hold(sym, "No price data", "", "", t, user_prompt=user_prompt)

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
                            t,
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
                        p.last_update = format_dt(t)
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
                            t,
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
                        p.last_update = format_dt(t)
                        if trade:
                            trade.update({"pnl": p.pnl, "equity": p.total_value})
                            trade_history.append(trade)

                    else:
                        p.hold(sym, reason or "HOLD", "", "", t, user_prompt=user_prompt)
                        p.last_update = format_dt(t)

        # Run all agents concurrently for this timestep
        await asyncio.gather(*(run_agent(n, p) for n, p in agent_states.items()))

        # Snapshot arena state for visualization
        history.append(
            {
                "timestamp": format_dt(t),
                "agents": {
                    n: {"pnl": p.pnl, "total_value": p.total_value}
                    for n, p in agent_states.items()
                },
            }
        )

        out = {
            "schema_version": "1.0",
            "updated_at": format_dt(t),
            "interval": INTERVAL,
            "agents": {n: p.to_dict() for n, p in agent_states.items()},
            "history": history[-HISTORY_LIMIT:],
            "trade_history": trade_history[-2000:],
        }
        atomic_save_json(STATE_FILE, out)
        print(f"💾 Step {step + 1}/{len(timestamps) - 50} | Trades={len(trade_history)}")

    print(f"\n✅ Backtest complete — {len(trade_history)} trades logged.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_backtest())
