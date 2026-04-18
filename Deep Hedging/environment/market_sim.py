"""
Market simulation and Black-Scholes analytics.

Provides:
  - Geometric Brownian Motion (GBM) stock price simulation
  - Black-Scholes European call price & delta
"""

import math
import torch
import numpy as np
from scipy.stats import norm as sp_norm


# ═══════════════════════════════════════════════════════════════════════
#  GBM Simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Simulate stock price paths under Geometric Brownian Motion.

        dS = μ·S·dt + σ·S·dW

    Uses the exact log-normal solution for accuracy:
        S_{t+1} = S_t · exp((μ - σ²/2)·dt + σ·√dt·Z)

    Args:
        S0:      Initial stock price
        mu:      Annualized drift
        sigma:   Annualized volatility
        T:       Time horizon in years
        n_steps: Number of discrete time steps
        n_paths: Number of simulated paths
        seed:    Optional random seed

    Returns:
        Tensor of shape (n_paths, n_steps + 1) with price paths.
        Column 0 is S0 for all paths.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * math.sqrt(dt)

    # Sample standard normal increments
    Z = torch.randn(n_paths, n_steps)
    log_returns = drift + diffusion * Z

    # Cumulative sum in log-space, prepend zero for S0
    log_prices = torch.cumsum(log_returns, dim=1)
    log_prices = torch.cat([torch.zeros(n_paths, 1), log_prices], dim=1)

    return S0 * torch.exp(log_prices)


# ═══════════════════════════════════════════════════════════════════════
#  Black-Scholes Analytics
# ═══════════════════════════════════════════════════════════════════════

def _d1(S: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (
        sigma * math.sqrt(tau)
    )


def bs_call_price(
    S: float, K: float, tau: float, sigma: float, r: float = 0.0
) -> float:
    """
    Black-Scholes European call price.

    Args:
        S:     Current stock price
        K:     Strike price
        tau:   Time to maturity (years)
        sigma: Volatility (annualized)
        r:     Risk-free rate (annualized)

    Returns:
        Call option price.
    """
    if tau <= 1e-12:
        return max(S - K, 0.0)
    d1 = _d1(S, K, tau, sigma, r)
    d2 = d1 - sigma * math.sqrt(tau)
    return float(
        S * sp_norm.cdf(d1) - K * math.exp(-r * tau) * sp_norm.cdf(d2)
    )


def bs_delta_call(
    S: float, K: float, tau: float, sigma: float, r: float = 0.0
) -> float:
    """
    Black-Scholes delta for a European call option.

    Δ = N(d₁)

    Args:
        S:     Current stock price
        K:     Strike price
        tau:   Time to maturity (years)
        sigma: Volatility (annualized)
        r:     Risk-free rate (annualized)

    Returns:
        Delta ∈ [0, 1].
    """
    if tau <= 1e-12:
        return 1.0 if S > K else 0.0
    d1 = _d1(S, K, tau, sigma, r)
    return float(sp_norm.cdf(d1))


def bs_delta_call_batch(
    S: np.ndarray | torch.Tensor,
    K: float,
    tau: np.ndarray | torch.Tensor,
    sigma: float,
    r: float = 0.0,
) -> np.ndarray:
    """
    Vectorized Black-Scholes call delta for arrays of (S, tau).

    Args:
        S:     Stock prices, shape (n,)
        K:     Strike price (scalar)
        tau:   Times to maturity, shape (n,)
        sigma: Volatility (scalar)
        r:     Risk-free rate (scalar)

    Returns:
        Deltas, shape (n,).
    """
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    delta = np.where(S > K, 1.0, 0.0)  # default for tau ≈ 0
    mask = tau > 1e-12

    if mask.any():
        d1 = (np.log(S[mask] / K) + (r + 0.5 * sigma ** 2) * tau[mask]) / (
            sigma * np.sqrt(tau[mask])
        )
        delta[mask] = sp_norm.cdf(d1)

    return delta
