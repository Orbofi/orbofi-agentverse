# ===============================================================
#  Flask Backend for Multi-Agent Arena Visualization — Full Detail
# ===============================================================

import os, json, time, requests
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

STATE_FILE = os.getenv("ARENA_STATE_FILE", "arena_state.json")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "schema_version": "1.0",
            "updated_at": None,
            "trading_mode": "paper",
            "interval": "15m",
            "max_order_usd": 15.0,
            "agents": {},
            "history": [],
            "trade_history": [],
        }

def normalize_agent(agent):
    cash = agent.get("cash", 0.0)
    positions = agent.get("positions") or agent.get("holdings") or {}
    total_value = agent.get("total_value", 0.0)
    pnl = agent.get("pnl", 0.0)

    if total_value <= 0:
        total_value = cash + sum(
            (pos.get("qty", 0.0) * pos.get("entry", 0.0)) for pos in positions.values()
        )
    if pnl == 0.0 and total_value > 0:
        pnl = total_value - 100000.0

    # Trade logs with full detail
    trade_log = []
    for t in agent.get("trade_log", []):
        trade_log.append({
            "timestamp": t.get("timestamp"),
            "type": t.get("type"),
            "symbol": t.get("symbol"),
            "price": t.get("price"),
            "amount": t.get("amount"),
            "qty": t.get("qty"),
            "reason": t.get("reason"),
            "reasoning": t.get("reasoning"), 
            "user_prompt": t.get("user_prompt", ""),
            "exit_strategy": t.get("exit_strategy", ""),
            "thinking": t.get("thinking") or t.get("thinking_log", []),
            "pnl_snapshot": t.get("pnl_snapshot", 0.0),
            "equity_snapshot": t.get("equity_snapshot", 0.0),
        })

    # Pass last_decision as-is (so we don't lose new fields)
    last_decision_raw = agent.get("last_decision") or {}
    if isinstance(last_decision_raw, str):
        try:
            last_decision_raw = json.loads(last_decision_raw)
        except Exception:
            last_decision_raw = {}

    return {
        "name": agent.get("name"),
        "id": agent.get("id"),
        "model": agent.get("model", "gpt-5"),
        "personality": agent.get("personality", ""),
        "cash": round(cash, 2),
        "holdings": positions,
        "positions": positions,
        "total_value": round(total_value, 2),
        "pnl": round(pnl, 2),
        "trades": list(reversed(trade_log)),
        "last_update": agent.get("last_update"),
        "last_decision": last_decision_raw,
    }

@app.route("/api/closed_trades", methods=["GET"])
def get_closed_trades():
    """
    Returns all trades marked as 'CLOSE' from arena_state.json.
    Includes full reasoning, user prompt, and exit details.
    """
    state = load_state()
    agents = state.get("agents", {})
    closed_trades = []

    for agent_name, agent_data in agents.items():
        img = agent_data.get("img", "")
        pnl = agent_data.get("pnl", 0.0)
        total_value = agent_data.get("total_value", 0.0)
        positions = agent_data.get("positions", {})
        last_decision = agent_data.get("last_decision", {})

        for t in agent_data.get("trade_log", []):
            act = str(t.get("action") or t.get("type") or "").upper()
            if act == "CLOSE":  # ✅ This matches your backtest logic
                closed_trades.append({
                    "agent": agent_name,
                    "agent_img": img,
                    "timestamp": t.get("timestamp"),
                    "action": act,
                    "symbol": t.get("symbol"),
                    "price": t.get("price"),
                    "qty": t.get("qty"),
                    "reason": t.get("reason"),
                    "reasoning": t.get("reasoning"),
                    "exit_strategy": (
                        t.get("exit_strategy")
                        or t.get("exit_reason")
                        or last_decision.get("exit_strategy_reason")
                    ),
                    "user_prompt": t.get("user_prompt", ""),
                    "thinking": t.get("thinking") or t.get("thinking_log", []),
                    "pnl_snapshot": t.get("pnl_snapshot", pnl),
                    "equity_snapshot": t.get("equity_snapshot", total_value),
                    "leverage": positions.get(t.get("symbol", ""), {}).get("leverage"),
                    "margin": positions.get(t.get("symbol", ""), {}).get("margin"),
                    "status": "CLOSED",
                })

    # sort descending by time
    closed_trades.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return jsonify(closed_trades)


@app.route("/api/open_trades", methods=["GET"])
def get_open_trades():
    state = load_state()
    agents = state.get("agents", {})
    open_agents = []

    for agent_name, agent_data in agents.items():
        positions = agent_data.get("positions", {})
        open_positions = {sym: pos for sym, pos in positions.items() if pos.get("qty", 0) > 0}

        # skip agents with no open positions
        if not open_positions:
            continue

        # rebuild full agent structure (same as /api/agents)
        open_agent = {
            "name": agent_name,
            "img": agent_data.get("img", ""),
            "pnl": agent_data.get("pnl", 0.0),
            "cash": agent_data.get("cash", 0.0),
            "total_value": agent_data.get("total_value", 0.0),
            "positions": open_positions,
            "holdings": open_positions,  # alias
            "last_decision": agent_data.get("last_decision", {}),
            "trade_log": agent_data.get("trade_log", []),
            "trading_mode": agent_data.get("trading_mode", "live"),
            "last_update": agent_data.get("last_update", datetime.utcnow().isoformat()),
            "status": "OPEN",
        }

        open_agents.append(open_agent)

    return jsonify(open_agents)



# ---------------------------------------------------------------
# Core API routes (agents, history, leaderboard, trades, meta)
# ---------------------------------------------------------------
@app.route("/api/agents", methods=["GET"])
def get_agents():
    state = load_state()
    agents = [normalize_agent(a) for a in state.get("agents", {}).values()]
    return jsonify(agents)

@app.route("/api/trades_data", methods=["GET"])
def get_trades():
    state = load_state()
    agents = state.get("agents", {})
    all_trades = []

    for agent_name, agent_data in agents.items():
        img = agent_data.get("img", "")
        pnl = agent_data.get("pnl", 0.0)
        cash = agent_data.get("cash", 0.0)
        total_value = agent_data.get("total_value", 0.0)
        positions = agent_data.get("positions", {})
        last_decision = agent_data.get("last_decision", {})

        # 🧩 Add completed trades
        # for t in agent_data.get("trade_log", []):
        #     all_trades.append({
        #         "agent": agent_name,
        #         "agent_img": img,
        #         "timestamp": t.get("timestamp"),
        #         "action": t.get("type") or t.get("action"),
        #         "symbol": t.get("symbol"),
        #         "reason": t.get("reason"),
        #         "reasoning": t.get("reasoning"),
        #         "exit_strategy": t.get("exit_strategy") or last_decision.get("exit_strategy_reason"),
        #         "user_prompt": t.get("user_prompt"),
        #         "thinking": t.get("thinking") or t.get("thinking_log"),
        #         "pnl_snapshot": t.get("pnl_snapshot", pnl),
        #         "equity_snapshot": t.get("equity_snapshot", total_value),
        #         "cash": cash,
        #         "positions": positions,
        #         "leverage": positions.get(t.get("symbol", ""), {}).get("leverage", None),
        #         "margin": positions.get(t.get("symbol", ""), {}).get("margin", None),
        #         "entry": positions.get(t.get("symbol", ""), {}).get("entry", None),
        #         "qty": positions.get(t.get("symbol", ""), {}).get("qty", None),
        #         "status": "CLOSED",
        #     })

        # 🧠 Add current open positions
        for sym, pos in positions.items():
            if pos.get("qty", 0) > 0:
                all_trades.append({
                    "agent": agent_name,
                    "agent_img": img,
                    "timestamp": agent_data.get("last_update") or datetime.utcnow().isoformat(),
                    "action": "OPEN",
                    "symbol": sym,
                    "price": pos.get("entry"),
                    "qty": pos.get("qty"),
                    "leverage": pos.get("leverage"),
                    "margin": pos.get("margin"),
                    "reason": last_decision.get("reason"),
                    "reasoning": last_decision.get("reasoning"),
                    "exit_strategy": last_decision.get("exit_strategy_reason"),
                    "user_prompt": last_decision.get("user_prompt"),
                    "pnl_snapshot": pnl,
                    "equity_snapshot": total_value,
                    "status": "OPEN",
                })

    # Sort everything by timestamp descending
    all_trades.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return jsonify(all_trades)



@app.route("/api/agent_image", methods=["GET"])
def get_agent_image():
    """
    Returns the agent's image URL as a string.
    Accepts ?name=<agent_name> or ?id=<agent_id>.
    Example:
      /api/agent_image?name=elon_musk
      /api/agent_image?id=003
    """
    name = request.args.get("name", "").strip().lower().replace(" ", "_")
    agent_id = request.args.get("id")
    state = load_state()
    agents = state.get("agents", {})

    for a in agents.values():
        # Normalize both for matching
        agent_name_norm = (a.get("name", "").strip().lower().replace(" ", "_"))
        if (name and agent_name_norm == name) or (agent_id and str(a.get("id")) == str(agent_id)):
            img = a.get("img") or a.get("last_decision", {}).get("agent_img", "")
            return jsonify(img or "")

    return jsonify("")


@app.route("/api/history", methods=["GET"])
def get_pnl_history():
    state = load_state()
    history = state.get("history", [])
    chart = {"timestamps": [], "agents": {}}

    if history:
        chart["timestamps"] = [snap["timestamp"] for snap in history]
        agent_names = set()
        for snap in history:
            agent_names.update(snap.get("agents", {}).keys())

        for name in agent_names:
            pnl_values = [
                snap.get("agents", {}).get(name, {}).get("pnl", 0)
                for snap in history
            ]
            start_balance = 100000.0
            chart["agents"][name] = [
                (v / start_balance * 100.0) if v is not None else 0
                for v in pnl_values
            ]

    return jsonify(chart)

@app.route("/api/current_pnl", methods=["GET"])
def current_pnl():
    state = load_state()
    agents = state.get("agents", {})
    chart = {"timestamps": [datetime.utcnow().isoformat()], "agents": {}}

    for name, data in agents.items():
        pnl = data.get("pnl", 0)
        chart["agents"][name] = [round(pnl, 2)]

    return jsonify(chart)

@app.route("/api/value_history", methods=["GET"])
def get_value_history():
    state = load_state()
    history = state.get("history", [])
    chart = {"timestamps": [], "agents": {}}

    if history:
        chart["timestamps"] = [snap["timestamp"] for snap in history]
        agent_names = set()
        for snap in history:
            agent_names.update(snap.get("agents", {}).keys())

        for name in agent_names:
            chart["agents"][name] = [
                snap.get("agents", {}).get(name, {}).get("total_value", None)
                for snap in history
            ]

    return jsonify(chart)

@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    try:
        with open("arena_leaderboard.json", "r") as f:
            board = json.load(f)
        return jsonify(board)
    except Exception:
        state = load_state()
        agents = [
            {
                "name": n,
                "model": a.get("model", "gpt-5"),
                "total_value": a.get("total_value", 0.0),
                "pnl": a.get("pnl", 0.0),
                "trades": len(a.get("trade_log", [])),
            }
            for n, a in state.get("agents", {}).items()
        ]

        agents.sort(key=lambda x: x["pnl"], reverse=True)
        return jsonify(agents)

@app.route("/api/agent_performance", methods=["GET"])
def agent_performance():
    state = load_state()
    agents_data = []
    for name, agent in state.get("agents", {}).items():
        agents_data.append({
            "name": name,
            "pnl": agent.get("pnl", 0.0),
            "equity": agent.get("total_value", 0.0),
            "trades": len(agent.get("trade_log", [])),
            "personality": agent.get("personality", "Neutral"),
        })
    return jsonify(agents_data)

@app.route("/api/trade_history", methods=["GET"])
def trade_history():
    state = load_state()
    combined = []
    for name, agent in state.get("agents", {}).items():
        for t in agent.get("trade_log", []):
            combined.append({
                "agent": name,
                "timestamp": t.get("timestamp"),
                "type": t.get("type"),
                "side": t.get("side"),
                "symbol": t.get("symbol"),
                "price": t.get("price"),
                "qty": t.get("qty"),
                "amount": t.get("amount"),
                "reason": t.get("reason"),
                "reasoning": t.get("reasoning"),
                "user_prompt": t.get("user_prompt", ""),
                "exit_strategy": t.get("exit_strategy", ""),
                "thinking": t.get("thinking") or t.get("thinking_log", []),
                "pnl": t.get("pnl", 0.0),
                "equity": t.get("equity", 0.0),
                "leverage": t.get("leverage", 0),
                "margin": t.get("amount_margin_usd"),
            })
    combined.sort(
        key=lambda t: (
            datetime.fromisoformat(t["timestamp"].replace("Z", ""))
            if t.get("timestamp")
            else datetime.min
        ),
        reverse=True,
    )
    return jsonify(combined)

@app.route("/api/meta", methods=["GET"])
def meta():
    state = load_state()
    return jsonify({
        "schema_version": state.get("schema_version", "1.0"),
        "updated_at": state.get("updated_at"),
        "trading_mode": state.get("trading_mode", "paper"),
        "interval": state.get("interval", "15m"),
        "max_order_usd": state.get("max_order_usd", 15.0),
        "agent_count": len(state.get("agents", {}))
    })

# ---------------------------------------------------------------
# 🌐 AsterDex Crypto Streaming Endpoints
# ---------------------------------------------------------------
ASTERDEX_API = "https://fapi.asterdex.com/fapi/v1"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"
]

# 🧠 Cache storage
_last_prices_cache = {
    "timestamp": 0,
    "data": [],
}

@app.route("/api/crypto_prices", methods=["GET"])
def get_crypto_prices():
    global _last_prices_cache

    try:
        r = requests.get(f"{ASTERDEX_API}/ticker/24hr", timeout=8)
        r.raise_for_status()
        data = r.json()
        filtered = [t for t in data if t["symbol"] in SYMBOLS]
        filtered.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)

        top6 = [
            {
                "symbol": t["symbol"],
                "price": float(t.get("lastPrice", 0)),
                "change_24h": float(t.get("priceChangePercent", 0)),
                "volume": float(t.get("volume", 0)),
            }
            for t in filtered[:6]
        ]

        # 🧠 Save successful result to cache
        _last_prices_cache = {
            "timestamp": time.time(),
            "data": top6,
        }

        return jsonify(top6)

    except Exception as e:
        print("⚠️ Error fetching Asterdex data:", e)

        # 🧩 Fallback to cached result
        if _last_prices_cache["data"]:
            print("⚙️ Serving cached crypto prices.")
            return jsonify({
                "cached": True,
                "timestamp": _last_prices_cache["timestamp"],
                "data": _last_prices_cache["data"],
            })

        # ❌ No cache yet, real error
        return jsonify({"error": str(e)}), 500
# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
