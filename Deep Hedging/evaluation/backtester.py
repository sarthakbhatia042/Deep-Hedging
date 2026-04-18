"""
Backtester: Compare deep hedging agent vs. Black-Scholes delta hedging.

Runs Monte Carlo simulations and evaluates PnL distributions for:
  1. No hedging (naked short call)
  2. Black-Scholes delta hedging
  3. Deep Hedging agent (CashInvariantDBH)
"""

import numpy as np
import torch
from environment.market_sim import simulate_gbm, bs_delta_call_batch
from evaluation.metrics import compute_risk_metrics


class Backtester:
    """
    Monte Carlo backtester for hedging strategies.

    Args:
        S0:               Initial stock price
        strike:           Option strike
        mu:               Drift
        sigma:            Volatility
        T:                Time to maturity (years)
        n_steps:          Number of rebalancing steps
        transaction_cost: Proportional cost rate
        risk_free_rate:   Risk-free rate
    """

    def __init__(
        self,
        S0: float = 100.0,
        strike: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        T: float = 1 / 12,
        n_steps: int = 30,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
    ):
        self.S0 = S0
        self.strike = strike
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.transaction_cost = transaction_cost
        self.r = risk_free_rate

    @classmethod
    def from_config(cls, config) -> "Backtester":
        """Create backtester from a Config dataclass."""
        return cls(
            S0=config.S0,
            strike=config.strike,
            mu=config.mu,
            sigma=config.sigma,
            T=config.T,
            n_steps=config.n_steps,
            transaction_cost=config.transaction_cost,
            risk_free_rate=config.risk_free_rate,
        )

    def run(
        self,
        model: torch.nn.Module | None = None,
        n_paths: int = 10000,
        seed: int = 123,
    ) -> dict:
        """
        Run the backtest.

        Args:
            model:   Trained CashInvariantDBH model (optional).
            n_paths: Number of Monte Carlo paths.
            seed:    Random seed for reproducibility.

        Returns:
            Dict with keys:
                "No Hedge":          PnL array
                "Black-Scholes":     PnL array
                "Deep Hedge":        PnL array (if model provided)
                "paths":             Price paths (n_paths, n_steps+1)
                "bs_deltas":         BS hedge ratios (n_paths, n_steps)
                "dh_deltas":         DH hedge ratios (n_paths, n_steps)
        """
        dt = self.T / self.n_steps

        # ── Simulate price paths ───────────────────────────────────────
        paths = simulate_gbm(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            T=self.T,
            n_steps=self.n_steps,
            n_paths=n_paths,
            seed=seed,
        )
        paths_np = paths.numpy()

        # Terminal payoff (we're SHORT the call)
        payoff = np.maximum(paths_np[:, -1] - self.strike, 0.0)

        # ── Strategy 1: No Hedging ─────────────────────────────────────
        pnl_no_hedge = -payoff

        # ── Strategy 2: Black-Scholes Delta Hedging ────────────────────
        bs_deltas = np.zeros((n_paths, self.n_steps))
        pnl_bs = np.zeros(n_paths)
        bs_position = np.zeros(n_paths)

        for t in range(self.n_steps):
            S_t = paths_np[:, t]
            tau_t = self.T - t * dt

            # BS delta
            delta_t = bs_delta_call_batch(S_t, self.strike, tau_t, self.sigma, self.r)
            bs_deltas[:, t] = delta_t

            # Transaction cost
            cost = self.transaction_cost * np.abs(delta_t - bs_position) * S_t
            pnl_bs -= cost

            # Hedge PnL: hold delta_t, stock moves to S_{t+1}
            S_next = paths_np[:, t + 1]
            pnl_bs += delta_t * (S_next - S_t)

            bs_position = delta_t

        # Subtract terminal payoff
        pnl_bs -= payoff

        results = {
            "No Hedge": pnl_no_hedge,
            "Black-Scholes": pnl_bs,
            "paths": paths_np,
            "bs_deltas": bs_deltas,
        }

        # ── Strategy 3: Deep Hedging ──────────────────────────────────
        if model is not None:
            model.eval()
            dh_deltas = np.zeros((n_paths, self.n_steps))
            pnl_dh = np.zeros(n_paths)
            dh_position = np.zeros(n_paths)

            with torch.no_grad():
                for t in range(self.n_steps):
                    S_t = paths_np[:, t]
                    tau_norm = 1.0 - t / self.n_steps

                    # Build state
                    state = torch.tensor(
                        np.stack(
                            [
                                np.log(S_t / self.strike),
                                np.full_like(S_t, tau_norm),
                                np.full_like(S_t, self.sigma),
                                dh_position,
                            ],
                            axis=-1,
                        ),
                        dtype=torch.float32,
                    )

                    # Get action from model
                    action = model.get_action(state).squeeze(-1).numpy()
                    dh_deltas[:, t] = action

                    # Transaction cost
                    cost = self.transaction_cost * np.abs(action - dh_position) * S_t
                    pnl_dh -= cost

                    # Hedge PnL
                    S_next = paths_np[:, t + 1]
                    pnl_dh += action * (S_next - S_t)

                    dh_position = action

            # Terminal payoff
            pnl_dh -= payoff

            results["Deep Hedge"] = pnl_dh
            results["dh_deltas"] = dh_deltas

        return results
