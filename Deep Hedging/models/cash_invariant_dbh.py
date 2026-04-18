"""
Cash-Invariant Deep Bellman Hedging (DBH) Agent.

Implements an Actor-Critic architecture for deep hedging with:
  - Entropic risk measure (cash-invariant Bellman target)
  - Q-function critic (state-action value)
  - Numerically stable LogSumExp trick

Reference:
  Buehler et al. (2019), "Deep Hedging"
  Murray et al. (2022), "Deep Hedging: Hedging Derivatives Under Generic Market Frictions"
"""

import torch
import torch.nn as nn


class CashInvariantDBH(nn.Module):
    """
    Cash-Invariant Deep Bellman Hedging agent.

    The actor network maps market states to hedge ratios in [0, 1].
    The critic network estimates the Q-value (risk-adjusted future cash flows)
    for a given state-action pair.

    Args:
        state_dim:     Dimension of the state vector.
        action_dim:    Dimension of the action vector.
        hidden_dim:    Width of hidden layers.
        risk_aversion: Entropic risk aversion parameter λ.
                       Higher λ → more risk-averse agent.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_dim: int = 128,
        risk_aversion: float = 1.0,
    ):
        super().__init__()
        self.lam = risk_aversion
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── ACTOR: State → Hedge Ratio ─────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # Hedge ratio ∈ [0, 1]
        )

        # ── Q-CRITIC: (State, Action) → Q-Value ───────────────────────
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the hedge ratio for the given state.

        Args:
            state: (batch, state_dim) tensor
        Returns:
            action: (batch, action_dim) hedge ratio ∈ [0, 1]
        """
        return self.actor(state)

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Estimate Q-value for a state-action pair.

        Args:
            state:  (batch, state_dim)
            action: (batch, action_dim)
        Returns:
            q: (batch, 1)
        """
        return self.critic(torch.cat([state, action], dim=-1))

    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        cash_flow: torch.Tensor,
        next_state: torch.Tensor,
        done: float,
    ):
        """
        Compute actor and critic losses.

        The critic is trained using the cash-invariant entropic risk Bellman
        equation. The actor maximizes the critic's Q-value estimate.

        Args:
            state:      (batch, state_dim)   current state
            action:     (batch, action_dim)  action taken (detached from actor)
            cash_flow:  (batch, 1)           immediate cash flow
            next_state: (batch, state_dim)   next state
            done:       float (0 or 1)       whether episode is finished

        Returns:
            actor_loss:  scalar tensor
            critic_loss: scalar tensor
        """
        # ── 1. Current Q-Value Estimate ────────────────────────────────
        q_curr = self.critic(torch.cat([state, action], dim=-1))

        # ── 2. Target Q-Value (no gradient) ────────────────────────────
        with torch.no_grad():
            next_action = self.actor(next_state)
            q_next = self.critic(torch.cat([next_state, next_action], dim=-1))

            # Cash-Invariant Bellman Target:
            #   Q_target = c(s,a) + (1 - done) · Q(s', π(s'))
            target_q = cash_flow + (1.0 - done) * q_next

        # ── 3. Critic Loss: Entropic Risk of TD-Error ──────────────────
        # Entropic risk = (1/λ) · log E[exp(-λ · δ)]
        # where δ = target_q - q_curr (TD-error)
        td_error = target_q - q_curr
        scaled_error = -self.lam * td_error

        # LogSumExp trick for numerical stability
        max_val = scaled_error.max().detach()
        critic_loss = (
            max_val
            + torch.log(torch.mean(torch.exp(scaled_error - max_val)))
        ) / self.lam

        # ── 4. Actor Loss: Maximize Q (minimize risk/liability) ────────
        current_actor_action = self.actor(state)
        actor_loss = -self.critic(
            torch.cat([state, current_actor_action], dim=-1)
        ).mean()

        return actor_loss, critic_loss

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "risk_aversion": self.lam,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "CashInvariantDBH":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        model = cls(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            risk_aversion=cfg["risk_aversion"],
            **kwargs,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
