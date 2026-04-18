"""
Batched RL environment for European call option hedging.

The agent is short one European call option and must dynamically
hedge using the underlying stock. At each step the agent chooses
a new hedge ratio (position in the stock as a fraction of one share).

State:  [log_moneyness, normalized_time_to_maturity, volatility, current_delta]
Action: New hedge ratio ∈ [0, 1]

Cash flows per step:
  cf_t = δ_t · (S_{t+1} - S_t) - cost · |δ_t - δ_{t-1}| · S_t

At the terminal step the option payoff is subtracted:
  cf_T += -max(S_T - K, 0)
"""

import torch
from environment.market_sim import simulate_gbm


class HedgingEnv:
    """
    Batched episodic hedging environment.

    All operations are vectorized over `batch_size` parallel paths
    for efficient GPU/CPU training.

    Args:
        S0:               Initial stock price
        strike:           Option strike price
        mu:               Drift (annualized)
        sigma:            Volatility (annualized)
        T:                Time to maturity (years)
        n_steps:          Number of rebalancing steps
        batch_size:       Number of parallel paths
        transaction_cost: Proportional cost rate
        seed:             Random seed (optional)
    """

    def __init__(
        self,
        S0: float = 100.0,
        strike: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        T: float = 1 / 12,
        n_steps: int = 30,
        batch_size: int = 512,
        transaction_cost: float = 0.001,
        seed: int | None = None,
    ):
        self.S0 = S0
        self.strike = strike
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.transaction_cost = transaction_cost
        self.seed = seed

        # Will be set in reset()
        self.paths: torch.Tensor | None = None
        self.step_idx: int = 0
        self.current_delta: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config) -> "HedgingEnv":
        """Create environment from a Config dataclass."""
        return cls(
            S0=config.S0,
            strike=config.strike,
            mu=config.mu,
            sigma=config.sigma,
            T=config.T,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            transaction_cost=config.transaction_cost,
            seed=config.seed,
        )

    def reset(self) -> torch.Tensor:
        """
        Reset the environment: simulate new price paths.

        Returns:
            Initial state, shape (batch_size, 4).
        """
        self.paths = simulate_gbm(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            T=self.T,
            n_steps=self.n_steps,
            n_paths=self.batch_size,
            seed=None,  # different paths each episode
        )
        self.step_idx = 0
        self.current_delta = torch.zeros(self.batch_size)
        return self._get_state()

    def step(self, action: torch.Tensor):
        """
        Take one hedging step.

        Args:
            action: (batch_size, 1) new hedge ratio ∈ [0, 1]

        Returns:
            next_state: (batch_size, 4)
            cash_flow:  (batch_size, 1)
            done:       bool
        """
        new_delta = action.squeeze(-1)  # (batch_size,)
        old_delta = self.current_delta

        S_curr = self.paths[:, self.step_idx]
        S_next = self.paths[:, self.step_idx + 1]

        # ── Cash flow components ───────────────────────────────────────
        # Hedge PnL from holding old_delta shares (old position earns this step's return)
        # Actually: at step t, the agent chose new_delta = δ_t
        # The hedge PnL is δ_t * (S_{t+1} - S_t) [the new position is held]
        hedge_pnl = new_delta * (S_next - S_curr)

        # Transaction cost for rebalancing
        rebalance_cost = (
            self.transaction_cost * torch.abs(new_delta - old_delta) * S_curr
        )

        cash_flow = hedge_pnl - rebalance_cost

        # Advance step
        self.step_idx += 1
        self.current_delta = new_delta.detach()

        done = self.step_idx >= self.n_steps

        if done:
            # Terminal: subtract option payoff (we're short the call)
            payoff = torch.clamp(S_next - self.strike, min=0.0)
            cash_flow = cash_flow - payoff

        next_state = self._get_state()
        return next_state, cash_flow.unsqueeze(-1), done

    def _get_state(self) -> torch.Tensor:
        """
        Construct the state vector.

        State = [log_moneyness, normalized_tau, sigma, current_delta]

        Returns:
            (batch_size, 4) tensor.
        """
        idx = min(self.step_idx, self.n_steps)
        S = self.paths[:, idx]

        log_moneyness = torch.log(S / self.strike)

        # Normalized time remaining: goes from 1 → 0
        tau_normalized = 1.0 - idx / self.n_steps

        state = torch.stack(
            [
                log_moneyness,
                torch.full_like(S, tau_normalized),
                torch.full_like(S, self.sigma),
                self.current_delta if self.current_delta is not None
                else torch.zeros_like(S),
            ],
            dim=-1,
        )
        return state

    def get_prices_at_step(self, step: int) -> torch.Tensor:
        """Get stock prices at a specific step."""
        return self.paths[:, step]

    def get_time_to_maturity(self, step: int) -> float:
        """Get time to maturity at a specific step (in years)."""
        return self.T * (1.0 - step / self.n_steps)
