import numpy as np

def asian_call_payoff(price_paths: np.ndarray, strike: float, r: float, T: float) -> np.ndarray:
    """
    Computes discounted payoff for an Asian call option (average price over path).

    Args:
        price_paths (np.ndarray): Simulated asset price paths, shape (M, N+1)
        strike (float): Strike price of the option
        r (float): Risk-free interest rate
        T (float): Time to maturity

    Returns:
        np.ndarray: Discounted Asian option payoffs, shape (M,)
    """
    average_prices = np.mean(price_paths[:, 1:], axis=1)  # exclude t=0
    payoffs = np.maximum(average_prices - strike, 0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    return discounted_payoffs


def barrier_call_payoff(price_paths: np.ndarray, strike: float, barrier: float, r: float, T: float) -> np.ndarray:
    """
    Computes discounted payoff for a knock-out barrier call option.

    The option pays (S_T - K)+ only if the max price during the path is below the barrier.

    Args:
        price_paths (np.ndarray): Simulated asset price paths, shape (M, N+1)
        strike (float): Strike price of the option
        barrier (float): Barrier level (knock-out)
        r (float): Risk-free interest rate
        T (float): Time to maturity

    Returns:
        np.ndarray: Discounted barrier option payoffs, shape (M,)
    """
    S_T = price_paths[:, -1]
    max_price = np.max(price_paths, axis=1)
    knock_out_mask = max_price < barrier
    payoffs = np.where(knock_out_mask, np.maximum(S_T - strike, 0), 0.0)
    discounted = np.exp(-r * T) * payoffs
    return discounted


def lookback_call_payoff(price_paths: np.ndarray, r: float, T: float) -> np.ndarray:
    """
    Computes discounted payoff for a lookback call option (based on the minimum price).

    Payoff is max(S_T - min(S), 0)

    Args:
        price_paths (np.ndarray): Simulated asset price paths, shape (M, N+1)
        r (float): Risk-free interest rate
        T (float): Time to maturity

    Returns:
        np.ndarray: Discounted lookback option payoffs, shape (M,)
    """
    S_T = price_paths[:, -1]
    S_min = np.min(price_paths, axis=1)
    payoffs = np.maximum(S_T - S_min, 0)
    discounted = np.exp(-r * T) * payoffs
    return discounted
