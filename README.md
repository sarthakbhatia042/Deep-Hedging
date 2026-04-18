# 📈 Deep Hedging — Cash-Invariant Deep Bellman Hedging

A reinforcement learning system that learns to optimally hedge European call options using a **Cash-Invariant Deep Bellman Hedging (DBH)** agent with an entropic risk measure. The trained neural network outperforms classical Black-Scholes delta hedging in the presence of transaction costs and discrete rebalancing.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)

---

## 🧠 What is Deep Hedging?

Traditional option hedging uses the **Black-Scholes delta** — a closed-form formula that assumes continuous trading and zero transaction costs. In reality:

- Markets have **transaction costs**
- Rebalancing happens at **discrete intervals**
- Volatility is **not constant**

**Deep Hedging** replaces the formula with a **neural network** that learns the optimal hedging strategy directly from simulated market data, accounting for all these real-world frictions.

### Key Innovation: Cash-Invariant Bellman Equation

Our agent uses the **entropic risk measure** instead of simple expected value:

$$
\rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\left[e^{-\lambda X}\right]
$$

This makes the agent **risk-averse** — it penalizes large losses more than it rewards equivalent gains, leading to more robust hedging strategies.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              CashInvariantDBH Agent              │
├─────────────────────┬───────────────────────────┤
│    ACTOR (Policy)   │    CRITIC (Q-Function)    │
│                     │                           │
│  State ────┐        │  State ──┐                │
│            ▼        │          ├──► Q-Value      │
│      [128 + LN]     │  Action ─┘                │
│         SiLU        │      [128 + LN]           │
│      [128 + LN]     │         SiLU              │
│         SiLU        │      [128 + LN]           │
│        [1]          │         SiLU              │
│       Sigmoid       │        [1]                │
│            ▼        │                           │
│     Hedge Ratio     │                           │
│       δ ∈ [0,1]     │                           │
└─────────────────────┴───────────────────────────┘

State = [log(S/K), τ_normalized, σ, δ_current]
Action = New hedge ratio ∈ [0, 1]
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd "Deep Hedging"

# Create virtual environment
python -m venv .dhenv
source .dhenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

This opens an **interactive web dashboard** where you can:
- 🎯 Train the agent with custom parameters
- 📊 Backtest against Black-Scholes delta hedging
- 🗺️ Visualize the learned hedge surface
- 📚 Read about the theory

### 3. Train via CLI

```bash
# Default training
python train.py

# Custom parameters
python train.py --episodes 5000 --sigma 0.3 --risk_aversion 2.0 --cost 0.002

# See all options
python train.py --help
```

---

## 📁 Project Structure

```
Deep Hedging/
├── app.py                       # 🖥️  Streamlit dashboard
├── train.py                     # ⌨️  CLI training entry point
├── requirements.txt             # 📦 Dependencies
│
├── models/
│   └── cash_invariant_dbh.py    # 🧠 Actor-Critic neural network
│
├── environment/
│   ├── market_sim.py            # 📈 GBM simulator + BS analytics
│   └── option_env.py            # 🎮 Batched RL hedging environment
│
├── training/
│   ├── config.py                # ⚙️  Hyperparameter configuration
│   └── trainer.py               # 🏋️ Training loop
│
├── evaluation/
│   ├── metrics.py               # 📏 Risk metrics (VaR, CVaR, Sharpe)
│   └── backtester.py            # 🔬 Monte Carlo backtester
│
└── checkpoints/                 # 💾 Saved model weights
```

---

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| **🎯 Train & Hedge** | Train the model with a real-time progress bar, view loss curves and single-path simulations |
| **📊 Backtest** | Monte Carlo comparison of No Hedge vs. Black-Scholes vs. Deep Hedge with PnL distributions |
| **🗺️ Hedge Surface** | Heatmaps showing how the agent hedges across (price, time) space vs. Black-Scholes |
| **📚 Theory** | Mathematical explanation of cash-invariant deep hedging |

---

## ⚙️ Configuration

Key parameters you can tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `S₀` | 100 | Initial stock price |
| `K` | 100 | Strike price |
| `σ` | 0.20 | Volatility (annualized) |
| `T` | 21 days | Time to maturity |
| `λ` | 1.0 | Risk aversion (higher = more conservative) |
| `cost` | 0.001 | Transaction cost rate |
| `episodes` | 2000 | Training episodes |
| `batch_size` | 512 | Paths per training batch |

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
