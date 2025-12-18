"""
Main Runner
Orchestrates portfolio optimization and stress testing, outputs JSON results.
"""

import json
import argparse
from datetime import datetime
from typing import List, Optional

from data_loader import get_clean_data
from portfolio_optimizer import optimize_portfolio, backtest_portfolio
from scenario_model import (
    StressScenario,
    create_trump_tariff_scenario,
    run_stress_simulation,
    THAI_ASSETS
)


# Default ticker universe
DEFAULT_TICKERS = [
    # Thai Export
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    # Thai Domestic
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    # Global
    'WDC', 'THD',
    # Fixed Income
    'LEMB', 'VWOB', 'EMLC',
    # FX
    'THB=X'
]


def run_analysis(
    tickers: List[str] = None,
    historical_years: int = 10,
    forward_days: int = 252,
    n_simulations: int = 1000,
    initial_drop_mean: float = -0.10,
    initial_drop_std: float = 0.02,
    volatility_multiplier: float = 1.10,
    affected_groups: List[str] = None,
    output_file: Optional[str] = None,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Run complete portfolio analysis with stress testing.

    Args:
        tickers: List of ticker symbols (default: DEFAULT_TICKERS)
        historical_years: Years of historical data (default: 10)
        forward_days: Days for forward simulation (default: 252 = 1 year)
        n_simulations: Number of Monte Carlo simulations (default: 1000)
        initial_drop_mean: Mean initial drop for stressed assets (default: -10%)
        initial_drop_std: Std dev of initial drop (default: +-2%)
        volatility_multiplier: Volatility multiplier (default: 1.10)
        affected_groups: Asset groups to stress (default: all Thai assets)
        output_file: Path to save JSON output (optional)
        risk_free_rate: Annual risk-free rate (default: 2%)

    Returns:
        Dictionary with complete analysis results
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if affected_groups is None:
        affected_groups = ['all']

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    prices, returns, removed_tickers = get_clean_data(
        tickers,
        years=historical_years,
        max_na_ratio=0.3,
        outlier_std=5.0
    )
    available_tickers = returns.columns.tolist()
    print(f"  - Loaded {len(available_tickers)} assets")
    print(f"  - Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  - Total observations: {len(returns)}")
    if removed_tickers:
        print(f"  - Removed tickers: {removed_tickers}")

    # Step 2: Optimize portfolio
    print("\n[2/4] Optimizing portfolio (Max Sharpe, No Shorting)...")
    opt_result = optimize_portfolio(
        returns,
        objective='max_sharpe',
        risk_free_rate=risk_free_rate
    )
    print(f"  - Expected Return: {opt_result['expected_return']:.2%}")
    print(f"  - Volatility: {opt_result['volatility']:.2%}")
    print(f"  - Sharpe Ratio: {opt_result['sharpe_ratio']:.2f}")

    # Show top weights
    sorted_weights = sorted(
        opt_result['weights'].items(),
        key=lambda x: -x[1]
    )
    print("  - Top holdings:")
    for ticker, weight in sorted_weights[:5]:
        if weight > 0.01:
            print(f"      {ticker}: {weight:.2%}")

    # Step 3: Backtest on historical data
    print("\n[3/4] Running historical backtest...")
    backtest_result = backtest_portfolio(returns, opt_result['weights'])
    print(f"  - Total Return: {backtest_result['total_return']:.2%}")
    print(f"  - Annualized Return: {backtest_result['annualized_return']:.2%}")
    print(f"  - Annualized Volatility: {backtest_result['annualized_volatility']:.2%}")
    print(f"  - Max Drawdown: {backtest_result['max_drawdown']:.2%}")

    # Step 4: Run stress scenario
    print("\n[4/4] Running stress scenario simulation...")
    scenario = create_trump_tariff_scenario(
        initial_drop_mean=initial_drop_mean,
        initial_drop_std=initial_drop_std,
        volatility_multiplier=volatility_multiplier,
        affected_groups=affected_groups,
        duration_days=forward_days,
        n_simulations=n_simulations
    )

    # Filter affected assets to only those available
    scenario.affected_assets = [
        a for a in scenario.affected_assets
        if a in available_tickers
    ]

    print(f"  - Scenario: {scenario.name}")
    print(f"  - Affected assets: {len(scenario.affected_assets)}")
    print(f"  - Initial drop: {scenario.initial_drop_mean:.1%} +/- {scenario.initial_drop_std:.1%}")
    print(f"  - Volatility multiplier: {scenario.volatility_multiplier:.2f}")
    print(f"  - Simulations: {scenario.n_simulations}")

    stress_result = run_stress_simulation(
        returns, prices, opt_result['weights'], scenario
    )

    print(f"\n  Stress Scenario Results:")
    print(f"  - Mean Return: {stress_result['statistics']['mean_return']:.2%}")
    print(f"  - Median Return: {stress_result['statistics']['median_return']:.2%}")
    print(f"  - VaR 95%: {stress_result['statistics']['var_95']:.2%}")
    print(f"  - CVaR 95%: {stress_result['statistics']['cvar_95']:.2%}")
    print(f"  - Probability of Loss: {stress_result['statistics']['prob_loss']:.1%}")

    # Compile final results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'historical_years': historical_years,
            'forward_days': forward_days,
            'n_simulations': n_simulations,
            'risk_free_rate': risk_free_rate,
            'available_tickers': available_tickers,
            'removed_tickers': removed_tickers
        },
        'optimal_weights': opt_result['weights'],
        'optimization_metrics': {
            'expected_return': opt_result['expected_return'],
            'volatility': opt_result['volatility'],
            'sharpe_ratio': opt_result['sharpe_ratio']
        },
        'backtest_results': {
            'total_return': backtest_result['total_return'],
            'annualized_return': backtest_result['annualized_return'],
            'annualized_volatility': backtest_result['annualized_volatility'],
            'sharpe_ratio': backtest_result['sharpe_ratio'],
            'max_drawdown': backtest_result['max_drawdown'],
            'start_date': backtest_result['start_date'],
            'end_date': backtest_result['end_date'],
            'daily_returns': backtest_result['portfolio_returns'],
            'cumulative_returns': backtest_result['cumulative_returns']
        },
        'stress_scenario': {
            'name': stress_result['scenario_name'],
            'parameters': stress_result['simulation_params'],
            'affected_assets': stress_result['affected_assets'],
            'initial_shocks': stress_result['initial_shocks'],
            'start_date': stress_result['start_date'],
            'end_date': stress_result['end_date'],
            'statistics': stress_result['statistics'],
            'percentiles': stress_result['percentiles'],
            'simulation_paths': stress_result['simulation_paths'],
            'all_simulations': stress_result['all_simulations']
        }
    }

    # Save to file if specified
    if output_file:
        # Convert datetime keys to strings for JSON serialization
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(i) for i in obj]
            else:
                return obj

        results_serializable = convert_keys(results)

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        print(f"\n[OUTPUT] Results saved to: {output_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )

    parser.add_argument(
        '--historical-years', type=int, default=10,
        help='Years of historical data (default: 10)'
    )
    parser.add_argument(
        '--forward-days', type=int, default=252,
        help='Days for forward simulation (default: 252)'
    )
    parser.add_argument(
        '--n-simulations', type=int, default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--initial-drop', type=float, default=-0.10,
        help='Mean initial drop for stressed assets (default: -0.10)'
    )
    parser.add_argument(
        '--drop-std', type=float, default=0.02,
        help='Std dev of initial drop (default: 0.02)'
    )
    parser.add_argument(
        '--vol-multiplier', type=float, default=1.10,
        help='Volatility multiplier for stressed assets (default: 1.10)'
    )
    parser.add_argument(
        '--affected-groups', type=str, nargs='+',
        default=['all'],
        choices=['export', 'domestic', 'etf', 'fx', 'all'],
        help='Asset groups to stress (default: all)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='portfolio_results.json',
        help='Output JSON file path (default: portfolio_results.json)'
    )
    parser.add_argument(
        '--risk-free-rate', type=float, default=0.02,
        help='Annual risk-free rate (default: 0.02)'
    )

    args = parser.parse_args()

    run_analysis(
        tickers=DEFAULT_TICKERS,
        historical_years=args.historical_years,
        forward_days=args.forward_days,
        n_simulations=args.n_simulations,
        initial_drop_mean=args.initial_drop,
        initial_drop_std=args.drop_std,
        volatility_multiplier=args.vol_multiplier,
        affected_groups=args.affected_groups,
        output_file=args.output,
        risk_free_rate=args.risk_free_rate
    )


if __name__ == "__main__":
    main()
