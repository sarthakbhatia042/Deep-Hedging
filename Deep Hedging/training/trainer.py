"""
Training loop for the Cash-Invariant Deep Bellman Hedging agent.

Supports:
  - Batched episodic training (vectorized over paths)
  - Separate actor/critic optimizers
  - Gradient clipping
  - Periodic logging and checkpoint saving
"""

import time
import torch
import torch.optim as optim
from models.cash_invariant_dbh import CashInvariantDBH
from environment.option_env import HedgingEnv
from training.config import Config


class Trainer:
    """
    Trains the CashInvariantDBH agent.

    Args:
        config: Config dataclass with all hyperparameters.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cpu")  # CPU is fine for this scale

        # ── Create model ────────────────────────────────────────────────
        self.model = CashInvariantDBH(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            risk_aversion=config.risk_aversion,
        ).to(self.device)

        # ── Create environment ──────────────────────────────────────────
        self.env = HedgingEnv.from_config(config)

        # ── Separate optimizers ─────────────────────────────────────────
        self.actor_optimizer = optim.Adam(
            self.model.actor.parameters(),
            lr=config.lr_actor,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = optim.Adam(
            self.model.critic.parameters(),
            lr=config.lr_critic,
            weight_decay=config.weight_decay,
        )

        # ── Training history ────────────────────────────────────────────
        self.history = {
            "actor_loss": [],
            "critic_loss": [],
            "mean_pnl": [],
            "std_pnl": [],
            "mean_delta": [],
        }

    def train(self, progress_callback=None) -> dict:
        """
        Run the full training loop.

        Args:
            progress_callback: Optional callable(episode, n_episodes, info_dict)
                               for progress bars (e.g., Streamlit).

        Returns:
            Training history dict.
        """
        config = self.config
        model = self.model
        model.train()

        start_time = time.time()

        for episode in range(config.n_episodes):
            # ── Reset environment ───────────────────────────────────────
            state = self.env.reset()

            episode_actor_loss = 0.0
            episode_critic_loss = 0.0
            episode_cash_flows = torch.zeros(config.batch_size)
            all_deltas = []

            # ── Run episode ─────────────────────────────────────────────
            for step in range(config.n_steps):
                # Get action (detach for critic update)
                action = model.get_action(state).detach()
                all_deltas.append(action.squeeze(-1).mean().item())

                # Step environment
                next_state, cash_flow, done = self.env.step(action)
                episode_cash_flows += cash_flow.squeeze(-1)

                # ── Step 1: Compute & update CRITIC ─────────────────────
                _, critic_loss = model.compute_loss(
                    state, action, cash_flow, next_state, float(done)
                )
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.critic.parameters(), config.grad_clip
                )
                self.critic_optimizer.step()

                # ── Step 2: Recompute & update ACTOR ────────────────────
                # Must recompute after critic update to avoid stale graph
                actor_loss, _ = model.compute_loss(
                    state, action, cash_flow, next_state, float(done)
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.actor.parameters(), config.grad_clip
                )
                self.actor_optimizer.step()

                episode_actor_loss += actor_loss.item()
                episode_critic_loss += critic_loss.item()
                state = next_state

            # ── Log episode metrics ─────────────────────────────────────
            avg_actor = episode_actor_loss / config.n_steps
            avg_critic = episode_critic_loss / config.n_steps
            mean_pnl = episode_cash_flows.mean().item()
            std_pnl = episode_cash_flows.std().item()
            mean_delta = sum(all_deltas) / len(all_deltas)

            self.history["actor_loss"].append(avg_actor)
            self.history["critic_loss"].append(avg_critic)
            self.history["mean_pnl"].append(mean_pnl)
            self.history["std_pnl"].append(std_pnl)
            self.history["mean_delta"].append(mean_delta)

            # ── Progress callback ───────────────────────────────────────
            if progress_callback is not None:
                info = {
                    "actor_loss": avg_actor,
                    "critic_loss": avg_critic,
                    "mean_pnl": mean_pnl,
                    "std_pnl": std_pnl,
                    "mean_delta": mean_delta,
                    "elapsed": time.time() - start_time,
                }
                progress_callback(episode, config.n_episodes, info)

            # ── Console logging ─────────────────────────────────────────
            if (episode + 1) % max(1, config.n_episodes // 10) == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Episode {episode + 1:>5}/{config.n_episodes} │ "
                    f"Actor: {avg_actor:+.4f} │ Critic: {avg_critic:.4f} │ "
                    f"PnL: {mean_pnl:+.3f} ± {std_pnl:.3f} │ "
                    f"δ̄: {mean_delta:.3f} │ {elapsed:.1f}s"
                )

        total_time = time.time() - start_time
        print(f"\n  ✓ Training complete in {total_time:.1f}s")

        return self.history
