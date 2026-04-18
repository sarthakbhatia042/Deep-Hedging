"""
Deep Hedging — Interactive Streamlit Dashboard

A premium interactive web application for training, visualizing,
and backtesting the Cash-Invariant Deep Bellman Hedging agent
against classical Black-Scholes delta hedging.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import torch
import pandas as pd

from models.cash_invariant_dbh import CashInvariantDBH
from environment.option_env import HedgingEnv
from environment.market_sim import bs_call_price, bs_delta_call
from training.config import Config
from training.trainer import Trainer
from evaluation.backtester import Backtester
from evaluation.metrics import compute_risk_metrics, compare_strategies

# ═══════════════════════════════════════════════════════════════════════
#  Page Configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Deep Hedging | Cash-Invariant DBH",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  Custom CSS for Premium Look
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 212, 170, 0.15);
}
.main-header h1 {
    color: #00D4AA;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94A3B8;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1A1F2E, #151923);
    border: 1px solid rgba(0, 212, 170, 0.12);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0, 212, 170, 0.1);
}
.metric-card .label {
    color: #64748B;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}
.metric-card .value {
    color: #F1F5F9;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.metric-card .value.positive { color: #00D4AA; }
.metric-card .value.negative { color: #FF6B6B; }

/* Section headers */
.section-header {
    color: #00D4AA;
    font-size: 1.1rem;
    font-weight: 600;
    padding-bottom: 0.5rem;
    margin-top: 1rem;
    border-bottom: 1px solid rgba(0, 212, 170, 0.2);
    letter-spacing: 0.5px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-trained {
    background: rgba(0, 212, 170, 0.15);
    color: #00D4AA;
    border: 1px solid rgba(0, 212, 170, 0.3);
}
.status-untrained {
    background: rgba(255, 107, 107, 0.15);
    color: #FF6B6B;
    border: 1px solid rgba(255, 107, 107, 0.3);
}

/* Info box */
.info-box {
    background: rgba(0, 212, 170, 0.05);
    border: 1px solid rgba(0, 212, 170, 0.15);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #94A3B8;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Theory math block */
.math-block {
    background: rgba(30, 41, 59, 0.8);
    border-left: 3px solid #00D4AA;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    font-family: 'Courier New', monospace;
    color: #E2E8F0;
    font-size: 0.95rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117, #141824);
    border-right: 1px solid rgba(0, 212, 170, 0.1);
}

/* Hide streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Plotly Theme
# ═══════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8"),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,212,170,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

COLORS = {
    "primary": "#00D4AA",
    "secondary": "#4CC9F0",
    "accent": "#F72585",
    "warning": "#FFB703",
    "no_hedge": "#FF6B6B",
    "bs": "#4CC9F0",
    "dh": "#00D4AA",
    "grid": "rgba(148, 163, 184, 0.1)",
}


# ═══════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def render_metric_card(label: str, value: str, css_class: str = ""):
    """Render a premium metric card."""
    st.markdown(
        f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value {css_class}">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def make_plotly_fig(**kwargs) -> go.Figure:
    """Create a Plotly figure with consistent styling."""
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>📈 Deep Hedging</h1>
    <p>Cash-Invariant Deep Bellman Hedging with Entropic Risk Measure — Train, visualize, and backtest a neural network that learns to hedge European call options.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="section-header">⚙️ MARKET PARAMETERS</div>', unsafe_allow_html=True)

    S0 = st.number_input("Initial Price (S₀)", value=100.0, min_value=10.0, max_value=500.0, step=5.0)
    strike = st.number_input("Strike Price (K)", value=100.0, min_value=10.0, max_value=500.0, step=5.0)
    sigma = st.slider("Volatility (σ)", min_value=0.05, max_value=0.80, value=0.20, step=0.01,
                       format="%.2f")
    T_days = st.slider("Maturity (trading days)", min_value=5, max_value=252, value=21, step=1)
    T = T_days / 252.0
    n_steps = st.slider("Rebalancing Steps", min_value=5, max_value=60, value=30, step=1)

    st.markdown('<div class="section-header">🧠 MODEL PARAMETERS</div>', unsafe_allow_html=True)

    risk_aversion = st.slider("Risk Aversion (λ)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                               format="%.1f")
    transaction_cost = st.slider("Transaction Cost", min_value=0.0, max_value=0.01, value=0.001,
                                  step=0.0001, format="%.4f")

    st.markdown('<div class="section-header">🏋️ TRAINING</div>', unsafe_allow_html=True)

    n_episodes = st.slider("Training Episodes", min_value=200, max_value=10000, value=2000, step=200)
    batch_size = st.select_slider("Batch Size", options=[128, 256, 512, 1024, 2048], value=512)
    learning_rate = st.select_slider("Learning Rate",
                                      options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
                                      value=1e-3, format_func=lambda x: f"{x:.0e}")

    st.markdown("---")

    # Option price info
    call_price = bs_call_price(S0, strike, T, sigma)
    moneyness = "ATM" if abs(S0 - strike) < 1 else ("ITM" if S0 > strike else "OTM")

    st.markdown(
        f"""<div class="info-box">
        <strong>BS Call Price:</strong> ${call_price:.2f}<br>
        <strong>Moneyness:</strong> {moneyness}<br>
        <strong>dt:</strong> {T/n_steps*252:.2f} trading days
        </div>""", unsafe_allow_html=True
    )

    # Model status
    trained = "model" in st.session_state and st.session_state["model"] is not None
    badge = "status-trained" if trained else "status-untrained"
    label = "✓ MODEL TRAINED" if trained else "✗ NOT TRAINED"
    st.markdown(f'<div class="status-badge {badge}">{label}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  Build Config
# ═══════════════════════════════════════════════════════════════════════

config = Config(
    S0=S0, strike=strike, mu=0.05, sigma=sigma, T=T,
    n_steps=n_steps, transaction_cost=transaction_cost,
    hidden_dim=128, risk_aversion=risk_aversion,
    n_episodes=n_episodes, batch_size=batch_size,
    lr_actor=learning_rate, lr_critic=learning_rate,
)


# ═══════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_train, tab_backtest, tab_surface, tab_theory = st.tabs([
    "🎯  Train & Hedge",
    "📊  Backtest",
    "🗺️  Hedge Surface",
    "📚  Theory",
])


# ─── TAB 1: Train & Hedge ──────────────────────────────────────────────
with tab_train:
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        train_clicked = st.button("Train Model", use_container_width=True, type="primary")

    if train_clicked:
        st.session_state["model"] = None
        st.session_state["history"] = None

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        def update_progress(ep, total, info):
            pct = (ep + 1) / total
            progress_bar.progress(
                pct,
                text=f"Episode {ep+1}/{total} — "
                     f"Actor: {info['actor_loss']:+.4f} | "
                     f"Critic: {info['critic_loss']:.4f} | "
                     f"PnL: {info['mean_pnl']:+.3f}"
            )

        trainer = Trainer(config)
        history = trainer.train(progress_callback=update_progress)

        st.session_state["model"] = trainer.model
        st.session_state["history"] = history
        st.session_state["config"] = config

        progress_bar.progress(1.0, text="✓ Training complete!")
        st.toast("🎉 Model trained successfully!", icon="✅")

    # ── Show training history ───────────────────────────────────────────
    if "history" in st.session_state and st.session_state["history"] is not None:
        history = st.session_state["history"]

        st.markdown("### 📈 Training Curves")
        fig_train = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss Convergence", "PnL over Training"),
            horizontal_spacing=0.08,
        )

        episodes = list(range(1, len(history["actor_loss"]) + 1))

        # Smooth the curves with rolling average
        def smooth(data, window=50):
            s = pd.Series(data)
            return s.rolling(window, min_periods=1).mean().tolist()

        fig_train.add_trace(go.Scatter(
            x=episodes, y=smooth(history["actor_loss"]),
            name="Actor Loss", line=dict(color=COLORS["primary"], width=2),
        ), row=1, col=1)
        fig_train.add_trace(go.Scatter(
            x=episodes, y=smooth(history["critic_loss"]),
            name="Critic Loss", line=dict(color=COLORS["accent"], width=2),
        ), row=1, col=1)
        fig_train.add_trace(go.Scatter(
            x=episodes, y=smooth(history["mean_pnl"]),
            name="Mean PnL", line=dict(color=COLORS["dh"], width=2),
            fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.08)",
        ), row=1, col=2)
        fig_train.add_trace(go.Scatter(
            x=episodes, y=smooth(history["std_pnl"]),
            name="Std PnL", line=dict(color=COLORS["warning"], width=2, dash="dot"),
        ), row=1, col=2)

        fig_train.update_layout(
            **PLOTLY_LAYOUT, height=380,
            showlegend=True,
        )
        fig_train.update_layout(
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        fig_train.update_xaxes(title_text="Episode", gridcolor=COLORS["grid"], zeroline=False)
        fig_train.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
        st.plotly_chart(fig_train, use_container_width=True)

    # ── Single Path Visualization ───────────────────────────────────────
    if "model" in st.session_state and st.session_state["model"] is not None:
        st.markdown("### 🔍 Single Path Simulation")

        model = st.session_state["model"]
        model.eval()

        # Simulate one path
        from environment.market_sim import simulate_gbm
        path = simulate_gbm(S0, 0.05, sigma, T, n_steps, 1, seed=None).squeeze().numpy()
        time_axis = np.linspace(0, T * 252, n_steps + 1)  # in trading days

        # Deep hedge along the path
        dh_deltas = []
        bs_deltas = []
        dh_pos = 0.0

        with torch.no_grad():
            for t in range(n_steps):
                tau_norm = 1.0 - t / n_steps
                tau_years = T * (1.0 - t / n_steps)
                state = torch.tensor(
                    [[np.log(path[t] / strike), tau_norm, sigma, dh_pos]],
                    dtype=torch.float32,
                )
                action = model.get_action(state).item()
                dh_deltas.append(action)
                dh_pos = action

                bs_d = bs_delta_call(path[t], strike, tau_years, sigma)
                bs_deltas.append(bs_d)

        # Plot: Price + Hedge Ratios
        fig_path = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Stock Price Path", "Hedge Ratio (Δ)"),
            row_heights=[0.55, 0.45],
        )

        fig_path.add_trace(go.Scatter(
            x=time_axis, y=path,
            name="Stock Price",
            line=dict(color="#E2E8F0", width=2.5),
            fill="tozeroy", fillcolor="rgba(226, 232, 240, 0.04)",
        ), row=1, col=1)

        # Strike line
        fig_path.add_hline(
            y=strike, line=dict(color=COLORS["warning"], width=1, dash="dash"),
            annotation_text=f"K = {strike}", row=1, col=1,
        )

        # Hedge ratios
        time_rebalance = time_axis[:-1]
        fig_path.add_trace(go.Scatter(
            x=time_rebalance, y=dh_deltas,
            name="Deep Hedge Δ",
            line=dict(color=COLORS["dh"], width=2.5),
        ), row=2, col=1)
        fig_path.add_trace(go.Scatter(
            x=time_rebalance, y=bs_deltas,
            name="BS Δ",
            line=dict(color=COLORS["bs"], width=2, dash="dot"),
        ), row=2, col=1)

        fig_path.update_layout(
            **PLOTLY_LAYOUT, height=520,
        )
        fig_path.update_layout(
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        )
        fig_path.update_xaxes(title_text="Trading Days", gridcolor=COLORS["grid"], zeroline=False)
        fig_path.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
        fig_path.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_path.update_yaxes(title_text="Delta", row=2, col=1)
        st.plotly_chart(fig_path, use_container_width=True)

        if st.button("🔄 Resimulate Path"):
            st.rerun()


# ─── TAB 2: Backtest ───────────────────────────────────────────────────
with tab_backtest:
    model_ready = "model" in st.session_state and st.session_state["model"] is not None

    if not model_ready:
        st.info("Train a model first in the **Train & Hedge** tab to see backtest results.")
    else:
        n_bt_paths = st.select_slider(
            "Number of Monte Carlo paths",
            options=[1000, 5000, 10000, 50000],
            value=10000,
        )

        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running Monte Carlo backtest..."):
                bt = Backtester.from_config(config)
                results = bt.run(
                    model=st.session_state["model"],
                    n_paths=n_bt_paths,
                    seed=42,
                )
                st.session_state["bt_results"] = results

        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]

            # ── Risk Metrics Table ──────────────────────────────────────
            st.markdown("### Risk Metrics Comparison")

            strategies = {}
            for name in ["No Hedge", "Black-Scholes", "Deep Hedge"]:
                if name in results:
                    strategies[name] = results[name]

            metrics_comparison = compare_strategies(strategies)
            df_metrics = pd.DataFrame(metrics_comparison).T
            df_metrics.index.name = "Strategy"

            # Style the table
            st.dataframe(
                df_metrics.style.format("{:.4f}").background_gradient(
                    cmap="RdYlGn", axis=0, subset=["Mean PnL", "Sharpe", "Sortino"]
                ).background_gradient(
                    cmap="RdYlGn_r", axis=0, subset=["Std PnL", "CVaR (95%)", "Max Loss"]
                ),
                use_container_width=True,
                height=160,
            )

            # ── Metric Cards ────────────────────────────────────────────
            dh_metrics = metrics_comparison.get("Deep Hedge", {})
            bs_metrics = metrics_comparison.get("Black-Scholes", {})

            col1, col2, col3, col4 = st.columns(4)

            dh_mean = dh_metrics.get("Mean PnL", 0)
            bs_mean = bs_metrics.get("Mean PnL", 0)
            improvement = ((dh_metrics.get("Std PnL", 1) - bs_metrics.get("Std PnL", 1))
                           / bs_metrics.get("Std PnL", 1) * 100) if bs_metrics.get("Std PnL", 1) else 0

            with col1:
                cls = "positive" if dh_mean >= 0 else "negative"
                render_metric_card("DH Mean PnL", f"${dh_mean:.3f}", cls)
            with col2:
                render_metric_card("DH Std PnL", f"${dh_metrics.get('Std PnL', 0):.3f}")
            with col3:
                render_metric_card("DH CVaR 95%", f"${dh_metrics.get('CVaR (95%)', 0):.3f}")
            with col4:
                cls = "positive" if improvement < 0 else "negative"
                render_metric_card("Risk vs BS", f"{improvement:+.1f}%", cls)

            st.markdown("")

            # ── PnL Distribution ────────────────────────────────────────
            st.markdown("###PnL Distributions")

            fig_pnl = make_plotly_fig(height=420)

            for name, color in [
                ("No Hedge", COLORS["no_hedge"]),
                ("Black-Scholes", COLORS["bs"]),
                ("Deep Hedge", COLORS["dh"]),
            ]:
                if name in results:
                    fig_pnl.add_trace(go.Histogram(
                        x=results[name],
                        name=name,
                        marker_color=color,
                        opacity=0.65,
                        nbinsx=80,
                    ))

            fig_pnl.update_layout(
                barmode="overlay",
                xaxis_title="Hedging PnL ($)",
                yaxis_title="Frequency",
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
            )
            fig_pnl.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
            st.plotly_chart(fig_pnl, use_container_width=True)

            # ── Cumulative Distribution ─────────────────────────────────
            st.markdown("###Cumulative PnL Distribution")

            fig_cdf = make_plotly_fig(height=380)

            for name, color in [
                ("No Hedge", COLORS["no_hedge"]),
                ("Black-Scholes", COLORS["bs"]),
                ("Deep Hedge", COLORS["dh"]),
            ]:
                if name in results:
                    sorted_pnl = np.sort(results[name])
                    cdf = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
                    fig_cdf.add_trace(go.Scatter(
                        x=sorted_pnl, y=cdf,
                        name=name,
                        line=dict(color=color, width=2.5),
                    ))

            fig_cdf.update_layout(
                xaxis_title="PnL ($)",
                yaxis_title="Cumulative Probability",
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
            )
            fig_cdf.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
            st.plotly_chart(fig_cdf, use_container_width=True)


# ─── TAB 3: Hedge Surface ──────────────────────────────────────────────
with tab_surface:
    model_ready = "model" in st.session_state and st.session_state["model"] is not None

    if not model_ready:
        st.info("Train a model first to visualize the learned hedge surface.")
    else:
        st.markdown("###Learned Hedge Ratio Surface")
        st.markdown(
            '<div class="info-box">This heatmap shows how the trained model hedges '
            'as a function of stock price and time to maturity, compared to Black-Scholes delta.</div>',
            unsafe_allow_html=True,
        )

        model = st.session_state["model"]
        model.eval()

        # Grid
        price_range = np.linspace(S0 * 0.75, S0 * 1.25, 60)
        tau_range = np.linspace(0.01, 1.0, 50)  # normalized tau

        dh_surface = np.zeros((len(tau_range), len(price_range)))
        bs_surface = np.zeros((len(tau_range), len(price_range)))

        with torch.no_grad():
            for i, tau_n in enumerate(tau_range):
                for j, S in enumerate(price_range):
                    state = torch.tensor(
                        [[np.log(S / strike), tau_n, sigma, 0.5]],
                        dtype=torch.float32,
                    )
                    dh_surface[i, j] = model.get_action(state).item()
                    bs_surface[i, j] = bs_delta_call(S, strike, tau_n * T, sigma)

        diff_surface = dh_surface - bs_surface

        col1, col2 = st.columns(2)

        with col1:
            fig_dh_surf = go.Figure(data=go.Heatmap(
                z=dh_surface,
                x=np.round(price_range, 1),
                y=np.round(tau_range, 2),
                colorscale="Viridis",
                colorbar=dict(title="Δ"),
                hovertemplate="Price: $%{x}<br>τ: %{y:.2f}<br>Δ: %{z:.3f}<extra>Deep Hedge</extra>",
            ))
            fig_dh_surf.update_layout(
                **PLOTLY_LAYOUT, height=420,
                title=dict(text="Deep Hedge Δ", font=dict(color=COLORS["primary"])),
                xaxis_title="Stock Price ($)",
                yaxis_title="Normalized Time to Maturity",
            )
            st.plotly_chart(fig_dh_surf, use_container_width=True)

        with col2:
            fig_bs_surf = go.Figure(data=go.Heatmap(
                z=bs_surface,
                x=np.round(price_range, 1),
                y=np.round(tau_range, 2),
                colorscale="Viridis",
                colorbar=dict(title="Δ"),
                hovertemplate="Price: $%{x}<br>τ: %{y:.2f}<br>Δ: %{z:.3f}<extra>Black-Scholes</extra>",
            ))
            fig_bs_surf.update_layout(
                **PLOTLY_LAYOUT, height=420,
                title=dict(text="Black-Scholes Δ", font=dict(color=COLORS["bs"])),
                xaxis_title="Stock Price ($)",
                yaxis_title="Normalized Time to Maturity",
            )
            st.plotly_chart(fig_bs_surf, use_container_width=True)

        # Difference surface
        st.markdown("### Hedge Ratio Difference (Deep Hedge − Black-Scholes)")

        fig_diff = go.Figure(data=go.Heatmap(
            z=diff_surface,
            x=np.round(price_range, 1),
            y=np.round(tau_range, 2),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="ΔDH − ΔBS"),
            hovertemplate="Price: $%{x}<br>τ: %{y:.2f}<br>Diff: %{z:.3f}<extra></extra>",
        ))
        fig_diff.update_layout(
            **PLOTLY_LAYOUT, height=400,
            xaxis_title="Stock Price ($)",
            yaxis_title="Normalized Time to Maturity",
        )
        st.plotly_chart(fig_diff, use_container_width=True)

        st.markdown(
            '<div class="info-box">'
            '<strong>Interpretation:</strong> Blue regions = DH hedges less aggressively than BS. '
            'Red regions = DH hedges more aggressively. Differences are typically due to the DH agent '
            'accounting for transaction costs (BS ignores them) and using the entropic risk measure '
            'instead of quadratic loss.'
            '</div>',
            unsafe_allow_html=True,
        )


# ─── TAB 4: Theory ─────────────────────────────────────────────────────
with tab_theory:
    st.markdown("### 📚 How Deep Hedging Works")

    st.markdown("""
<div class="info-box">
The <strong>Deep Hedging</strong> framework replaces the classical Black-Scholes delta-hedging
approach with a data-driven neural network that learns optimal hedging strategies directly
from simulated market data, accounting for realistic market frictions like transaction costs.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### 🎯 The Hedging Problem")
    st.markdown(
        "An option trader **sells** a European call option and receives the premium. "
        "They must then **dynamically trade the underlying stock** to offset the option's risk. "
        "The goal: minimize the variability of the total profit and loss."
    )

    st.markdown("#### 📐 Mathematical Formulation")

    st.latex(r"""
    \text{PnL} = \underbrace{\sum_{t=0}^{N-1} \delta_t \cdot (S_{t+1} - S_t)}_{\text{Hedge Gains}}
    - \underbrace{\sum_{t=0}^{N-1} c \cdot |\delta_t - \delta_{t-1}| \cdot S_t}_{\text{Transaction Costs}}
    - \underbrace{\max(S_T - K, 0)}_{\text{Option Payoff}}
    """)

    st.markdown("#### 🧠 Cash-Invariant Bellman Equation")
    st.markdown(
        "Unlike standard RL which uses expected value, we use the **entropic risk measure** "
        "which penalizes downside risk more heavily:"
    )

    st.latex(r"""
    \rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\left[e^{-\lambda X}\right]
    """)

    st.markdown("The **cash-invariant** Bellman equation becomes:")

    st.latex(r"""
    Q(s, a) = c(s, a) + Q\big(s', \pi(s')\big)
    """)

    st.markdown(
        "where $c(s,a)$ is the immediate cash flow. This is superior to the additive value formulation "
        "because it respects the **cash-invariance** property: adding a constant cash amount "
        "shifts the risk by exactly that amount."
    )

    st.markdown("#### 🏗️ Architecture")

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("**Actor Network (Policy)**")
        st.markdown(
            "- Input: State `[log(S/K), τ, σ, δ_curr]`\n"
            "- 3 layers with LayerNorm + SiLU\n"
            "- Output: Sigmoid → hedge ratio ∈ [0, 1]\n"
            "- Trained to **maximize Q-value**"
        )
    with col_arch2:
        st.markdown("**Critic Network (Q-Function)**")
        st.markdown(
            "- Input: State + Action concatenated\n"
            "- 3 layers with LayerNorm + SiLU\n"
            "- Output: Risk-adjusted Q-value\n"
            "- Trained with **entropic risk Bellman loss**"
        )

    st.markdown("#### 🔬 Why Deep Hedging Beats Black-Scholes")

    comparison_data = {
        "Feature": [
            "Transaction Costs",
            "Risk Measure",
            "Model Assumptions",
            "Adaptability",
            "Discrete Rebalancing",
        ],
        "Black-Scholes": [
            "❌ Ignored",
            "Quadratic (variance)",
            "GBM, continuous trading",
            "Fixed formula",
            "Suboptimal (designed for continuous)",
        ],
        "Deep Hedging": [
            "✅ Directly incorporated",
            "Entropic (tail-risk aware)",
            "Any simulated dynamics",
            "Learns from data",
            "✅ Optimized for discrete steps",
        ],
    }
    st.table(pd.DataFrame(comparison_data).set_index("Feature"))

    st.markdown("####References")
    st.markdown(
        "1. Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). "
        "*Deep Hedging*. Quantitative Finance.\n"
        "2. Murray, P., Wood, B., Buehler, H., Wiese, M., & Pham, M. (2022). "
        "*Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using "
        "Reinforcement Learning*. Swiss Finance Institute."
    )
