"""
Hyperparameters and configuration for the Deep Hedging training pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All hyperparameters for the Deep Hedging system."""

    # ── Market Parameters ──────────────────────────────────────────────
    S0: float = 100.0           # Initial stock price
    strike: float = 100.0       # Option strike price (ATM by default)
    mu: float = 0.05            # Drift (annualized)
    sigma: float = 0.2          # Volatility (annualized)
    risk_free_rate: float = 0.0 # Risk-free rate
    T: float = 1.0 / 12        # Time to maturity in years (1 month)
    n_steps: int = 30           # Number of rebalancing steps

    # ── Transaction Costs ──────────────────────────────────────────────
    transaction_cost: float = 0.001  # Proportional transaction cost rate

    # ── Model Parameters ──────────────────────────────────────────────
    state_dim: int = 4          # [log_moneyness, tau, sigma, delta]
    action_dim: int = 1         # Hedge ratio
    hidden_dim: int = 128       # Hidden layer width
    risk_aversion: float = 1.0  # Entropic risk aversion λ

    # ── Training Parameters ────────────────────────────────────────────
    n_episodes: int = 2000      # Number of training episodes
    batch_size: int = 512       # Paths per episode (batch)
    lr_actor: float = 1e-3      # Actor learning rate
    lr_critic: float = 1e-3     # Critic learning rate
    weight_decay: float = 1e-5  # L2 regularization
    grad_clip: float = 1.0      # Gradient clipping norm
    seed: int = 42              # Random seed

    # ── Evaluation ─────────────────────────────────────────────────────
    n_eval_paths: int = 10000   # Number of paths for evaluation

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.n_steps

    def describe(self) -> str:
        """Human-readable summary of the configuration."""
        lines = [
            "═══ Deep Hedging Configuration ═══",
            f"  Market:  S₀={self.S0}, K={self.strike}, σ={self.sigma}, "
            f"μ={self.mu}, T={self.T:.4f}y ({self.T*252:.0f} trading days)",
            f"  Hedging: {self.n_steps} rebalances, "
            f"cost={self.transaction_cost:.4f}",
            f"  Model:   hidden={self.hidden_dim}, λ={self.risk_aversion}",
            f"  Train:   {self.n_episodes} episodes × {self.batch_size} paths, "
            f"lr={self.lr_actor}",
        ]
        return "\n".join(lines)
