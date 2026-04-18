#!/usr/bin/env python3
"""
CLI entry point for training the Deep Hedging agent.

Usage:
    python train.py
    python train.py --episodes 5000 --risk_aversion 2.0 --sigma 0.3
    python train.py --save checkpoints/model.pt
"""

import argparse
import os
from training.config import Config
from training.trainer import Trainer
from evaluation.backtester import Backtester
from evaluation.metrics import compute_risk_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Cash-Invariant Deep Bellman Hedging Agent"
    )

    # Market parameters
    parser.add_argument("--S0", type=float, default=100.0, help="Initial price")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--mu", type=float, default=0.05, help="Drift")
    parser.add_argument("--T", type=float, default=1/12, help="Maturity (years)")
    parser.add_argument("--n_steps", type=int, default=30, help="Rebalancing steps")

    # Model parameters
    parser.add_argument("--risk_aversion", type=float, default=1.0, help="λ")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument("--save", type=str, default="checkpoints/model.pt",
                        help="Checkpoint save path")
    parser.add_argument("--eval_paths", type=int, default=10000,
                        help="Number of eval paths for backtest")

    args = parser.parse_args()

    # ── Build config ────────────────────────────────────────────────────
    config = Config(
        S0=args.S0,
        strike=args.strike,
        mu=args.mu,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        transaction_cost=args.cost,
        hidden_dim=args.hidden_dim,
        risk_aversion=args.risk_aversion,
        n_episodes=args.episodes,
        batch_size=args.batch_size,
        lr_actor=args.lr,
        lr_critic=args.lr,
        seed=args.seed,
    )

    print("\n" + config.describe() + "\n")

    # ── Train ───────────────────────────────────────────────────────────
    print("─── Training ─────────────────────────────────────────────")
    trainer = Trainer(config)
    trainer.train()

    # ── Save checkpoint ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    trainer.model.save(args.save)
    print(f"\n  💾 Model saved to {args.save}")

    # ── Backtest ────────────────────────────────────────────────────────
    print("\n─── Backtesting ──────────────────────────────────────────")
    bt = Backtester.from_config(config)
    results = bt.run(model=trainer.model, n_paths=args.eval_paths)

    for strategy in ["No Hedge", "Black-Scholes", "Deep Hedge"]:
        if strategy in results:
            metrics = compute_risk_metrics(results[strategy])
            print(f"\n  {strategy}:")
            for k, v in metrics.items():
                print(f"    {k:>15s}: {v:>10.4f}")

    print("\n  ✓ Done!\n")


if __name__ == "__main__":
    main()
