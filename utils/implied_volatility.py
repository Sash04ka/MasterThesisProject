import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call_price(S, K, T, sigma, r):
    """
    Black-Scholes price for a European call option.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility_call(C, S, K, T, r, tol=1e-6):
    """
    Estimate implied volatility for a European call option using Brent's method.

    Parameters:
    - C: observed call price
    - S: spot price
    - K: strike price
    - T: time to maturity (in years)
    - r: risk-free rate
    - tol: numerical tolerance for root finding

    Returns:
    - Implied volatility (float) or np.nan if it cannot be computed
    """
    if T <= 0 or C < max(S - K * np.exp(-r * T), 0):
        return np.nan

    try:
        return brentq(
            lambda sigma: black_scholes_call_price(S, K, T, sigma, r) - C,
            a=1e-4,
            b=5.0,
            xtol=tol
        )
    except (ValueError, RuntimeError):
        return np.nan
