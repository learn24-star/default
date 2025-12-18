"""
Scenario Modeling Module
Stress testing with customizable scenarios using Geometric Brownian Motion (GBM).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta


@dataclass
class StressScenario:
    """
    Configuration for a stress scenario.

    Attributes:
        name: Scenario name
        affected_assets: List of tickers affected by the stress
        initial_drop_mean: Mean initial price drop (e.g., -0.10 for -10%)
        initial_drop_std: Standard deviation of initial drop (e.g., 0.02 for +-2%)
        volatility_multiplier: Factor to multiply volatility (e.g., 1.10 for +10%)
        start_date: When the stress scenario starts (default: today)
        duration_days: Duration of forward simulation in trading days (default: 252 = 1 year)
        n_simulations: Number of Monte Carlo simulations (default: 1000)
    """
    name: str
    affected_assets: List[str]
    initial_drop_mean: float = -0.10
    initial_drop_std: float = 0.02
    volatility_multiplier: float = 1.10
    start_date: Optional[datetime] = None
    duration_days: int = 252
    n_simulations: int = 1000

    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now()


# Predefined Thai-related assets
THAI_ASSETS = {
    'export': ['DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK'],
    'domestic': ['CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK'],
    'etf': ['THD'],
    'fx': ['THB=X'],
    'all': [
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        'THD', 'THB=X'
    ]
}


def create_trump_tariff_scenario(
    initial_drop_mean: float = -0.10,
    initial_drop_std: float = 0.02,
    volatility_multiplier: float = 1.10,
    affected_groups: List[str] = None,
    duration_days: int = 252,
    n_simulations: int = 1000
) -> StressScenario:
    """
    Create a Trump tariffs stress scenario.

    Args:
        initial_drop_mean: Mean initial drop for Thai assets (default -10%)
        initial_drop_std: Std dev of initial drop (default +-2%)
        volatility_multiplier: Volatility increase factor (default +10%)
        affected_groups: List of asset groups to affect ('export', 'domestic', 'etf', 'fx', 'all')
        duration_days: Forward simulation duration
        n_simulations: Number of Monte Carlo paths

    Returns:
        StressScenario configuration
    """
    if affected_groups is None:
        affected_groups = ['all']

    affected_assets = []
    for group in affected_groups:
        if group in THAI_ASSETS:
            affected_assets.extend(THAI_ASSETS[group])

    # Remove duplicates while preserving order
    affected_assets = list(dict.fromkeys(affected_assets))

    return StressScenario(
        name="Trump Tariffs Impact",
        affected_assets=affected_assets,
        initial_drop_mean=initial_drop_mean,
        initial_drop_std=initial_drop_std,
        volatility_multiplier=volatility_multiplier,
        duration_days=duration_days,
        n_simulations=n_simulations
    )


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: int,
    n_paths: int,
    dt: float = 1/252
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.

    Args:
        S0: Initial price
        mu: Drift (annualized)
        sigma: Volatility (annualized)
        T: Number of time steps
        n_paths: Number of simulation paths
        dt: Time step (default: 1/252 for daily)

    Returns:
        Array of shape (n_paths, T+1) with simulated prices
    """
    # Generate random increments
    dW = np.random.standard_normal((n_paths, T)) * np.sqrt(dt)

    # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW

    # Cumulative sum to get log returns
    log_returns = drift + diffusion
    log_prices = np.cumsum(log_returns, axis=1)

    # Convert back to prices
    prices = np.zeros((n_paths, T + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(log_prices)

    return prices


def run_stress_simulation(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario
) -> Dict:
    """
    Run stress scenario simulation.

    Args:
        returns: Historical returns DataFrame
        prices: Historical prices DataFrame
        weights: Portfolio weights dictionary
        scenario: StressScenario configuration

    Returns:
        Dictionary with simulation results
    """
    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    T = scenario.duration_days
    n_sims = scenario.n_simulations

    # Calculate historical statistics (annualized)
    hist_mean = returns.mean() * 252
    hist_vol = returns.std() * np.sqrt(252)

    # Get last prices as starting point
    last_prices = prices.iloc[-1].to_dict()

    # Initialize arrays for all assets
    simulated_prices = {}
    initial_shocks = {}

    for ticker in tickers:
        S0 = last_prices[ticker]
        mu = hist_mean[ticker]
        sigma = hist_vol[ticker]

        # Check if asset is affected by stress scenario
        if ticker in scenario.affected_assets:
            # Apply initial shock
            shock = np.random.normal(
                scenario.initial_drop_mean,
                scenario.initial_drop_std,
                n_sims
            )
            initial_shocks[ticker] = shock.mean()

            # Shocked starting price
            S0_shocked = S0 * (1 + shock)

            # Increased volatility
            sigma_stressed = sigma * scenario.volatility_multiplier

            # Simulate each path with its own shocked starting price
            paths = np.zeros((n_sims, T + 1))
            for i in range(n_sims):
                paths[i:i+1, :] = simulate_gbm(
                    S0_shocked[i], mu, sigma_stressed, T, 1
                )
            simulated_prices[ticker] = paths
        else:
            # Normal simulation (no stress)
            initial_shocks[ticker] = 0.0
            simulated_prices[ticker] = simulate_gbm(S0, mu, sigma, T, n_sims)

    # Calculate portfolio value across simulations
    weights_array = np.array([weights.get(t, 0.0) for t in tickers])

    # Normalize weights (number of units of each asset to hold)
    initial_portfolio_value = 1.0
    units = {}
    for ticker in tickers:
        price = last_prices[ticker]
        weight = weights.get(ticker, 0.0)
        units[ticker] = (initial_portfolio_value * weight) / price

    # Calculate portfolio value at each time step for each simulation
    portfolio_values = np.zeros((n_sims, T + 1))

    for t in range(T + 1):
        for i, ticker in enumerate(tickers):
            portfolio_values[:, t] += units[ticker] * simulated_prices[ticker][:, t]

    # Calculate returns from portfolio values
    portfolio_returns = portfolio_values[:, 1:] / portfolio_values[:, :-1] - 1

    # Calculate statistics
    final_values = portfolio_values[:, -1]
    total_returns = final_values / initial_portfolio_value - 1

    # Percentile-based VaR and CVaR
    var_95 = np.percentile(total_returns, 5)
    var_99 = np.percentile(total_returns, 1)
    cvar_95 = total_returns[total_returns <= var_95].mean()
    cvar_99 = total_returns[total_returns <= var_99].mean()

    # Generate date index for results
    start = scenario.start_date
    date_range = pd.bdate_range(start=start, periods=T + 1)

    # Summary statistics
    results = {
        'scenario_name': scenario.name,
        'affected_assets': scenario.affected_assets,
        'initial_shocks': initial_shocks,
        'simulation_params': {
            'n_simulations': n_sims,
            'duration_days': T,
            'initial_drop_mean': scenario.initial_drop_mean,
            'initial_drop_std': scenario.initial_drop_std,
            'volatility_multiplier': scenario.volatility_multiplier
        },
        'start_date': str(start.date()),
        'end_date': str(date_range[-1].date()),
        'statistics': {
            'mean_return': float(total_returns.mean()),
            'median_return': float(np.median(total_returns)),
            'std_return': float(total_returns.std()),
            'min_return': float(total_returns.min()),
            'max_return': float(total_returns.max()),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'prob_loss': float((total_returns < 0).mean()),
            'prob_loss_10pct': float((total_returns < -0.10).mean()),
            'prob_loss_20pct': float((total_returns < -0.20).mean())
        },
        'percentiles': {
            str(p): float(np.percentile(total_returns, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        'simulation_paths': {
            'dates': [str(d.date()) for d in date_range],
            'mean_path': portfolio_values.mean(axis=0).tolist(),
            'median_path': np.median(portfolio_values, axis=0).tolist(),
            'percentile_5': np.percentile(portfolio_values, 5, axis=0).tolist(),
            'percentile_95': np.percentile(portfolio_values, 95, axis=0).tolist()
        },
        # Raw simulation data (all paths, downsampled for storage)
        'all_simulations': {
            'final_returns': total_returns.tolist(),
            'final_values': final_values.tolist()
        }
    }

    return results


def compare_scenarios(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    weights: Dict[str, float],
    scenarios: List[StressScenario]
) -> Dict:
    """
    Compare multiple stress scenarios.

    Args:
        returns: Historical returns DataFrame
        prices: Historical prices DataFrame
        weights: Portfolio weights dictionary
        scenarios: List of StressScenario configurations

    Returns:
        Dictionary with comparison results
    """
    results = {}
    for scenario in scenarios:
        print(f"Running scenario: {scenario.name}...")
        results[scenario.name] = run_stress_simulation(
            returns, prices, weights, scenario
        )

    # Create comparison summary
    comparison = {
        'scenario_names': [s.name for s in scenarios],
        'mean_returns': {
            name: res['statistics']['mean_return']
            for name, res in results.items()
        },
        'var_95': {
            name: res['statistics']['var_95']
            for name, res in results.items()
        },
        'prob_loss': {
            name: res['statistics']['prob_loss']
            for name, res in results.items()
        }
    }

    return {
        'scenarios': results,
        'comparison': comparison
    }


if __name__ == "__main__":
    # Test the module
    from data_loader import get_clean_data
    from portfolio_optimizer import optimize_portfolio

    tickers = [
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        'WDC', 'THD', 'LEMB', 'VWOB', 'EMLC', 'THB=X'
    ]

    # Load data
    prices, returns, removed = get_clean_data(tickers, years=10)

    # Optimize portfolio
    opt_result = optimize_portfolio(returns, objective='max_sharpe')
    weights = opt_result['weights']

    # Create stress scenario
    scenario = create_trump_tariff_scenario(
        initial_drop_mean=-0.10,
        initial_drop_std=0.02,
        volatility_multiplier=1.10,
        affected_groups=['all'],
        n_simulations=100  # Reduced for testing
    )

    # Run simulation
    print("\nRunning stress simulation...")
    results = run_stress_simulation(returns, prices, weights, scenario)

    print(f"\n=== {results['scenario_name']} Results ===")
    print(f"Mean Return: {results['statistics']['mean_return']:.2%}")
    print(f"Median Return: {results['statistics']['median_return']:.2%}")
    print(f"VaR 95%: {results['statistics']['var_95']:.2%}")
    print(f"CVaR 95%: {results['statistics']['cvar_95']:.2%}")
    print(f"Probability of Loss: {results['statistics']['prob_loss']:.1%}")
