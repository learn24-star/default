"""
Portfolio Optimizer Module
Mean-variance optimization using scipy with no-shorting constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, List


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Calculate portfolio expected return."""
    return np.dot(weights, mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio (annualized)."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free_rate) / vol


def neg_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """Negative Sharpe ratio for minimization."""
    return -sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'max_sharpe',
    risk_free_rate: float = 0.02,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None
) -> Dict:
    """
    Optimize portfolio weights using mean-variance optimization.

    Args:
        returns: DataFrame of daily returns
        objective: 'max_sharpe', 'min_volatility', 'target_return', or 'target_volatility'
        risk_free_rate: Annual risk-free rate (default 2%)
        target_return: Target annual return (for 'target_return' objective)
        target_volatility: Target annual volatility (for 'target_volatility' objective)

    Returns:
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    tickers = returns.columns.tolist()

    # Annualize statistics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Convert to numpy for optimization
    mean_ret_np = mean_returns.values
    cov_mat_np = cov_matrix.values

    # Initial guess (equal weights)
    init_weights = np.array([1.0 / n_assets] * n_assets)

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # Bounds: no shorting (weights between 0 and 1)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    # Define objective function based on type
    if objective == 'max_sharpe':
        obj_func = lambda w: neg_sharpe_ratio(w, mean_ret_np, cov_mat_np, risk_free_rate)

    elif objective == 'min_volatility':
        obj_func = lambda w: portfolio_volatility(w, cov_mat_np)

    elif objective == 'target_return':
        if target_return is None:
            raise ValueError("target_return must be specified for 'target_return' objective")
        obj_func = lambda w: portfolio_volatility(w, cov_mat_np)
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_return(w, mean_ret_np) - target_return
        })

    elif objective == 'target_volatility':
        if target_volatility is None:
            raise ValueError("target_volatility must be specified for 'target_volatility' objective")
        obj_func = lambda w: -portfolio_return(w, mean_ret_np)
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_volatility(w, cov_mat_np) - target_volatility
        })

    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Optimize
    result = minimize(
        obj_func,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    optimal_weights = result.x

    # Calculate portfolio metrics
    port_return = portfolio_return(optimal_weights, mean_ret_np)
    port_vol = portfolio_volatility(optimal_weights, cov_mat_np)
    port_sharpe = sharpe_ratio(optimal_weights, mean_ret_np, cov_mat_np, risk_free_rate)

    # Create weights dictionary
    weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)}

    return {
        'weights': weights_dict,
        'expected_return': float(port_return),
        'volatility': float(port_vol),
        'sharpe_ratio': float(port_sharpe),
        'optimization_success': result.success,
        'optimization_message': result.message
    }


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate the efficient frontier.

    Args:
        returns: DataFrame of daily returns
        n_points: Number of points on the frontier
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with efficient frontier points
    """
    # Find min and max return achievable
    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()

    target_returns = np.linspace(min_ret * 0.9, max_ret * 1.1, n_points)

    frontier_points = []

    for target_ret in target_returns:
        try:
            result = optimize_portfolio(
                returns,
                objective='target_return',
                target_return=target_ret,
                risk_free_rate=risk_free_rate
            )
            if result['optimization_success']:
                frontier_points.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio']
                })
        except Exception:
            continue

    return pd.DataFrame(frontier_points)


def backtest_portfolio(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 1.0
) -> Dict:
    """
    Backtest a portfolio with given weights.

    Args:
        returns: DataFrame of daily returns
        weights: Dictionary of {ticker: weight}
        initial_value: Initial portfolio value

    Returns:
        Dictionary with backtest results
    """
    # Align weights with returns columns
    tickers = returns.columns.tolist()
    weights_array = np.array([weights.get(t, 0.0) for t in tickers])

    # Calculate daily portfolio returns
    portfolio_returns = (returns * weights_array).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() * initial_value

    # Calculate metrics
    total_return = (cumulative_returns.iloc[-1] / initial_value) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'portfolio_returns': portfolio_returns.to_dict(),
        'cumulative_returns': cumulative_returns.to_dict(),
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'start_date': str(returns.index[0].date()),
        'end_date': str(returns.index[-1].date())
    }


if __name__ == "__main__":
    # Test the module
    from data_loader import get_clean_data

    tickers = [
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        'WDC', 'THD', 'LEMB', 'VWOB', 'EMLC', 'THB=X'
    ]

    prices, returns, removed = get_clean_data(tickers, years=10)

    # Optimize for maximum Sharpe ratio
    result = optimize_portfolio(returns, objective='max_sharpe')

    print("\n=== Optimal Portfolio (Max Sharpe) ===")
    print(f"Expected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for ticker, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.01:
            print(f"  {ticker}: {weight:.2%}")

    # Backtest
    backtest = backtest_portfolio(returns, result['weights'])
    print(f"\n=== Backtest Results ===")
    print(f"Total Return: {backtest['total_return']:.2%}")
    print(f"Annualized Return: {backtest['annualized_return']:.2%}")
    print(f"Max Drawdown: {backtest['max_drawdown']:.2%}")
