"""
Risk and performance metrics for evaluating hedging strategies.
"""

import numpy as np


def compute_risk_metrics(pnl: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Compute a comprehensive set of risk metrics for a PnL distribution.

    Args:
        pnl:        1-D array of PnL values (one per simulated path)
        confidence: Confidence level for VaR/CVaR (default 0.95)

    Returns:
        Dictionary of metric_name → value.
    """
    pnl = np.asarray(pnl, dtype=np.float64)

    mean = float(np.mean(pnl))
    std = float(np.std(pnl))

    # ── Value at Risk (VaR) ────────────────────────────────────────────
    # VaR_α = -quantile(pnl, 1-α)
    # Represents the worst loss at the given confidence level
    alpha = 1.0 - confidence
    var = float(-np.quantile(pnl, alpha))

    # ── Conditional VaR (Expected Shortfall / CVaR) ────────────────────
    # Average loss in the worst (1-α) fraction of scenarios
    tail = pnl[pnl <= -var] if var > 0 else pnl[pnl <= np.quantile(pnl, alpha)]
    cvar = float(-np.mean(tail)) if len(tail) > 0 else var

    # ── Max Drawdown (worst single-path loss) ──────────────────────────
    max_loss = float(-np.min(pnl))

    # ── Sharpe-like ratio ──────────────────────────────────────────────
    # mean / std, treating PnL as returns
    sharpe = mean / std if std > 1e-10 else 0.0

    # ── Semi-deviation (downside risk) ─────────────────────────────────
    downside = pnl[pnl < mean]
    semi_std = float(np.sqrt(np.mean((downside - mean) ** 2))) if len(downside) > 0 else 0.0

    # ── Sortino-like ratio ─────────────────────────────────────────────
    sortino = mean / semi_std if semi_std > 1e-10 else 0.0

    return {
        "Mean PnL": round(mean, 4),
        "Std PnL": round(std, 4),
        f"VaR ({confidence:.0%})": round(var, 4),
        f"CVaR ({confidence:.0%})": round(cvar, 4),
        "Max Loss": round(max_loss, 4),
        "Sharpe": round(sharpe, 4),
        "Semi-Std": round(semi_std, 4),
        "Sortino": round(sortino, 4),
    }


def compare_strategies(results: dict[str, np.ndarray], confidence: float = 0.95) -> dict:
    """
    Compare risk metrics across multiple hedging strategies.

    Args:
        results:    Dict of {strategy_name: pnl_array}
        confidence: Confidence level for VaR/CVaR

    Returns:
        Dict of {strategy_name: metrics_dict}
    """
    return {
        name: compute_risk_metrics(pnl, confidence)
        for name, pnl in results.items()
    }
