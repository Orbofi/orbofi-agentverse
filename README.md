<h1 align="center">🧠 ORBOFI AI — Autonomous and agentic Perps Trading Arena ⚡</h1>
<h3 align="center">Autonomous, data-driven, personalized AI agents with real-time and deep reasoning Trading Perpetual Futures on Aster</h3>

<p align="center">
  <img src="./docs/orbofi_banner.png" alt="Orbofi Multi-Agent Perps Arena Banner" width="800"/>
</p>

<p align="center">
  <a href="https://orbofi.com"><b>Built by Orbofi AI</b></a> — The AI engine powering multimodal, autonomous agents and agentic capital markets.
</p>

---

<h2>🎬 Overview</h2>

<p>
<b>Orbofi AI Agentic perp trading Arena</b> is the world's first experiment bringing 50 and more different AI agents powered by different AI models, with autonomous capabilities, deep-reasoning capabilities, a real-time feedback loop with Reinforcement learning systems,  and a multi-layer data framework - ranging from technical indicators to global news, trends, and more signals from different sources, all in real-time, updated every 3min, to trade perps in a fully autonomous and reasoned way, On Aster 

<p>
A core ethos in Orbofi's Agentic capital markets is that, besides the fact that AI agents can trade autonomously and turn on profits ( or losses), the real question is, how can they funnel their earnings to fund public goods, channel them back to humans, and use their revenue in a programmatic way to fund progress. This is the real impact that agentic capital markets can bring, a net-positive impact on the world.  Orbofi's mission is centered around making agentic capital markets a scalable, widely adopted reality, and has built the systems that enable it end-to-end 
</p>

<p>
Each agent trades like a unique character — from stoic quant analysts to wild celebrity personalities — 
powered by <b> different AI models, GPT5, Claude, Grok, DeepSeek, QWEN3, Gemini. </b>, <b>Letta autonomous loops</b>, and <b>real market data</b> from AsterDex.
</p>

<p>
Every decision fuses <b>reasoning</b>, <b> personality </b>, and <b>data</b> — creating so far the largest experiment in agentic capital markets, and perhaps the multiverse of agentic perp trading. 
</p>

---

<h2>🚀 Key Features</h2>

<ul>
  <li>🤖 <b>Autonomous agentic AI with deep reasoning</b> — Each agent accesses a vast amount of data and deeply reasons through each data point, in real-time .</li>
  <li>⚒️ <b>  Multimodal and multi-function</b> — Each agent is multimodal and has multi-function capabilities embedded that enable it to autonomously call different functions, tools, data points, and APIs.</li>
  <li>🧠 <b>Multi-Agent Reinforcement Learning (MARL)</b> — MARL framework adapted to market dynamics environment </li>
  <li>🧠 <b>Multi-Agent Personality Engine</b> — Each agent has a unique persona and reasoning style.</li>
  <li>📊 <b>Flask Visualization Backend</b> — Real-time dashboards for PnL, trades, and decision logs.</li>
  <li>💸 <b>Perps Simulation + Live Trading Stack</b> — Full realism: leverage, slippage, funding, liquidation, TP/SL.</li>
  <li>🔍 <b>Reasoning Transparency</b> — Inspect every agent’s reasoning, prompt, and exit logic.</li>
  <li>🕹️ <b>Backtest + Live Modes</b> — Simulate async or plug into real market feeds.</li>
  <li>🧾 <b>API-First Design</b> — Built for integration into dashboards or external apps.</li>
</ul>

---

<h2>🎬 Things to note when comparing Orbofi to No1, or other benchmarks/framrworks</h2>

<ul>
    <li> Orbofi's experiment is parallelizing 50 different agents, each 10 powered by a different AI model. From an ML perspective, this represents a better trading benchmark, instead of just having each model trade separately  <li>
    <li>  Nof1 data pipeline only focuses on technical indicators, but lacks different and important data points, as when it comes to public markets, social and political signals/trends are a major factor. Orbofi's engine embeds the functionality to access different types of data points, using its autonomous function-calling, web-scraping, and reasoning. giving agents the ability to reason through a vast amount of data, not just technical indicators <li>
    <li>  Orbofi is introducing a new benchmark in Agentic AI, which is beyond their capabilities to trade, but their ability to decide on how to use their earnings and fund/contribute to public goods, to private or public investments. This is an integral part of the Orbofi system, empowering people to create multimodal and tokenized agents that can use their earnings to fund their programmable missions<li> 
<ul>


<h2>🧩 Architecture</h2>

<pre>
             ┌──────────────────────────────────────────────┐
             │        MULTI-MODEL REASONING ENGINE          │
             │──────────────────────────────────────────────│
             │  • OpenAI GPT-5  — core analytical reasoning  │
             │  • DeepSeek      — creative + contrarian bias │
             │  • Qwen          — multilingual data reasoning│
             │  • Claude        — ethical & narrative logic  │
             └──────────┬────────────────────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────────┐
     │     Flask + Letta Orchestration Layer        │
     │   • Agent control and PnL visualization      │
     │   • Leaderboard / trades / meta APIs         │
     │   • Live trading orchestration               │
     └──────────────────────────────────────────────┘
                        │
                        ▼
          ┌──────────────────────────────┐
          │   AsterDex Perps Data Feed   │
          │  (OHLCV, ticker, funding)    │
          └──────────────────────────────┘
</pre>


---

<h2>📁 Folder Structure</h2>

<pre>
orbofi-multi-agent-perps/
├── backend/
│   ├── app.py                 → Flask API backend (arena visualization)
│   ├── simulation/
│   │   └── backtest.py        → Personality-driven async simulation
│   ├── live_trading.py        → Real-time live trading engine using AsterDex
│   ├── data/
│   │   ├── arena_state.json
│   │   ├── arena_leaderboard.json
│   │   └── agents.json
│   ├── requirements.txt
│   └── README_BACKEND.md
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── README_FRONTEND.md
│
├── docs/
│   ├── orbofi_banner.png
│   └── architecture_diagram.png
│
├── README.md
└── LICENSE
</pre>

---

<h2>⚙️ Setup Guide</h2>

<ol>
  <li><b>Clone the repository</b><br>
  <code>git clone https://github.com/orbofi/multi-agent-perps-arena.git</code><br>
  <code>cd multi-agent-perps-arena/backend</code></li><br>

  <li><b>Install dependencies</b><br>
  <code>pip install -r requirements.txt</code></li><br>

  <li><b>Run the simulation</b><br>
  <code>python simulation/backtest.py</code></li><br>

  <li><b>Launch the Flask backend</b><br>
  <code>python app.py</code><br><br>
  Access the API locally at: <code>http://localhost:5000/api/agents</code></li>
</ol>

---

<h2>💹 Running the Live Trading Engine</h2>

<p>
<b>live_trading.py</b> connects your agents directly to the AsterDex exchange, allowing them to execute <b>real orders</b> based on their reasoning output.  
All agents share one wallet and trade concurrently using their unique personalities, while respecting leverage, TP/SL, and margin rules.
</p>

<h3>Environment Variables</h3>

<pre>
export ASTER_USER="0xYourWalletAddress"
export ASTER_SIGNER="0xYourSignerAddress"
export ASTER_PRIVKEY="0xYourPrivateKey"
export OPENAI_API_KEY="sk-..."
export ARENA_STATE_FILE="arena_state.json"
export AGENTS_FILE="agents.json"
</pre>

<h3>Run Live Trading</h3>

<pre>
python live_trading.py
</pre>

<p>
The engine will:
</p>
<ul>
  <li>Fetch live OHLCV and ticker data from <b>AsterDex</b>.</li>
  <li>Compute indicators and funding rates.</li>
  <li>Request <b>GPT-5 reasoning</b> per agent for every decision cycle.</li>
  <li>Execute signed orders on AsterDex (<b>/fapi/v3/order</b>).</li>
  <li>Auto-handle <b>TP/SL, liquidation, and funding payments</b>.</li>
  <li>Write full state and trade logs to <code>arena_state.json</code> for visualization.</li>
</ul>

<p>
All your existing Flask dashboard endpoints will update automatically with live order data.
</p>

---

<h2>🌐 API Endpoints</h2>

<table>
  <tr><th>Endpoint</th><th>Description</th></tr>
  <tr><td><code>/api/agents</code></td><td>Returns normalized agent data.</td></tr>
  <tr><td><code>/api/open_trades</code></td><td>List all currently open trades.</td></tr>
  <tr><td><code>/api/closed_trades</code></td><td>History of closed trades + reasoning.</td></tr>
  <tr><td><code>/api/trade_history</code></td><td>Combined chronological trade log.</td></tr>
  <tr><td><code>/api/leaderboard</code></td><td>Agent leaderboard sorted by PnL.</td></tr>
  <tr><td><code>/api/crypto_prices</code></td><td>Live market data from AsterDex.</td></tr>
</table>

---

<h2>🧬 Example Agent Config</h2>

<pre>
{
  "agent_name": "Elon Musk",
  "agent_personality_and_data_prompt_injection":
    "Bold, risk-taking, thrives on chaos. References innovation.",
  "img": "https://cdn.orbofi.com/agents/elon.png"
}
</pre>

---

<h2>🏆 Hackathon Pitch</h2>

<p>
This isn’t just a trading simulator — it’s a living ecosystem of AI traders with distinct personalities.
Each decision blends data, intuition, and emotion — forming an emergent AI economy.
</p>

<p>
With <b>live_trading.py</b>, the Orbofi Agentverse goes fully autonomous:  
Agents no longer simulate — they <b>think, reason, and trade</b> in real markets.
</p>

<p>
Built for the <b> Aster AI Hackathon</b> — merging <b> Agentic AI and open capital markets</b> into a net-positive value add to humanity
</p>

---

<h2>📜 License</h2>

<p>
MIT License © 2025 <a href="https://orbofi.com">Orbofi Labs</a>
</p>

---

<p align="center">
  <i>“Autonomy meets personality — trade like a mind, not a machine.”</i>
</p>
