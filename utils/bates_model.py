import numpy as np
from config import S0, v0

def simulate_bates_paths(
    S0=S0,
    v0=0.04,
    r=0.0,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7,
    lambda_jump=0.1,
    mu_jump=-0.05,
    sigma_jump=0.2,
    T=1.0,
    N=252,
    M=10000,
    seed=None
):
    """
    Simulates asset price and variance paths using the Bates model with jumps and stochastic volatility.

    Args:
        S0 (float): Initial asset price
        v0 (float): Initial variance
        r (float): Risk-free interest rate
        kappa (float): Mean reversion speed of variance
        theta (float): Long-term mean of variance
        sigma_v (float): Volatility of variance (vol-of-vol)
        rho (float): Correlation between asset and variance shocks
        lambda_jump (float): Jump intensity (Poisson process)
        mu_jump (float): Mean jump size
        sigma_jump (float): Standard deviation of jump size
        T (float): Time to maturity
        N (int): Number of time steps
        M (int): Number of simulation paths
        seed (int or None): Random seed for reproducibility

    Returns:
        S (np.ndarray): Simulated asset price paths of shape (M, N+1)
        v (np.ndarray): Simulated variance paths of shape (M, N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, N + 1):
        z1 = np.random.normal(size=M)
        z2 = np.random.normal(size=M)

        # Correlated Brownian motions
        dW_S = z1
        dW_v = rho * z1 + np.sqrt(1 - rho**2) * z2

        # Variance process (Heston dynamics)
        v[:, t] = np.abs(
            v[:, t - 1]
            + kappa * (theta - v[:, t - 1]) * dt
            + sigma_v * np.sqrt(np.maximum(v[:, t - 1], 0) * dt) * dW_v
        )

        # Jump component
        jumps = np.random.poisson(lambda_jump * dt, M)
        jump_sizes = np.random.normal(mu_jump, sigma_jump, M)

        # Asset price dynamics with stochastic volatility and jumps
        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * v[:, t - 1]) * dt
            + np.sqrt(np.maximum(v[:, t - 1], 0) * dt) * dW_S
            + jumps * jump_sizes
        )

    return S, v
