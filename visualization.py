"""
Visualization Module
Creates charts and plots from portfolio optimization results.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# Use non-interactive backend for saving plots
plt.switch_backend('Agg')


def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_optimal_weights(results: Dict, output_path: str) -> None:
    """
    Create bar chart of optimal portfolio weights.
    """
    weights = results['optimal_weights']

    # Filter out zero weights and sort by value
    weights_filtered = {k: v for k, v in weights.items() if v > 0.001}
    weights_sorted = dict(sorted(weights_filtered.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(12, 6))

    tickers = list(weights_sorted.keys())
    values = list(weights_sorted.values())

    # Color by asset type
    colors = []
    for ticker in tickers:
        if ticker.endswith('.BK'):
            colors.append('#2ecc71')  # Green for Thai stocks
        elif ticker in ['LEMB', 'VWOB', 'EMLC']:
            colors.append('#3498db')  # Blue for fixed income
        elif ticker == 'THB=X':
            colors.append('#9b59b6')  # Purple for FX
        else:
            colors.append('#e74c3c')  # Red for global

    bars = ax.bar(tickers, values, color=colors, edgecolor='white', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Weight', fontsize=12)
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_title('Optimal Portfolio Weights (Max Sharpe, No Shorting)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Thai Stocks'),
        Patch(facecolor='#3498db', label='Fixed Income'),
        Patch(facecolor='#e74c3c', label='Global'),
        Patch(facecolor='#9b59b6', label='FX')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def parse_date(date_str: str) -> datetime:
    """Parse date string with flexible format handling."""
    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Fallback: try parsing just the date part
    return datetime.strptime(date_str[:10], '%Y-%m-%d')


def plot_backtest_performance(results: Dict, output_path: str) -> None:
    """
    Create line chart of historical backtest cumulative returns.
    """
    backtest = results['backtest_results']
    cumulative = backtest['cumulative_returns']

    # Convert string dates to datetime
    dates = [parse_date(d) for d in cumulative.keys()]
    values = list(cumulative.values())

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, values, color='#2980b9', linewidth=1.5, label='Portfolio Value')
    ax.fill_between(dates, 1, values, alpha=0.3, color='#3498db')

    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Add metrics annotation
    metrics_text = (
        f"Total Return: {backtest['total_return']:.1%}\n"
        f"Ann. Return: {backtest['annualized_return']:.1%}\n"
        f"Ann. Volatility: {backtest['annualized_volatility']:.1%}\n"
        f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {backtest['max_drawdown']:.1%}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Portfolio Value (Starting = 1.0)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f"Historical Backtest Performance ({backtest['start_date']} to {backtest['end_date']})",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_stress_simulation_paths(results: Dict, output_path: str) -> None:
    """
    Create fan chart of stress scenario simulation paths.
    """
    stress = results['stress_scenario']
    paths = stress['simulation_paths']

    dates = [parse_date(d) for d in paths['dates']]
    mean_path = paths['mean_path']
    median_path = paths['median_path']
    p5 = paths['percentile_5']
    p95 = paths['percentile_95']

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot confidence bands
    ax.fill_between(dates, p5, p95, alpha=0.3, color='#e74c3c', label='5th-95th Percentile')

    # Plot mean and median
    ax.plot(dates, mean_path, color='#c0392b', linewidth=2, label='Mean Path')
    ax.plot(dates, median_path, color='#e74c3c', linewidth=2, linestyle='--', label='Median Path')

    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Add metrics annotation
    stats = stress['statistics']
    metrics_text = (
        f"Scenario: {stress['name']}\n"
        f"Mean Return: {stats['mean_return']:.1%}\n"
        f"Median Return: {stats['median_return']:.1%}\n"
        f"VaR 95%: {stats['var_95']:.1%}\n"
        f"CVaR 95%: {stats['cvar_95']:.1%}\n"
        f"P(Loss): {stats['prob_loss']:.1%}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Portfolio Value (Starting = 1.0)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f"Stress Scenario: Forward Simulation ({stress['start_date']} to {stress['end_date']})",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_return_distribution(results: Dict, output_path: str) -> None:
    """
    Create histogram of simulated final returns.
    """
    stress = results['stress_scenario']
    final_returns = stress['all_simulations']['final_returns']
    stats = stress['statistics']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create histogram
    n, bins, patches = ax.hist(final_returns, bins=50, density=True,
                                color='#3498db', edgecolor='white', alpha=0.7)

    # Color bars based on return
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if right < 0:
            patch.set_facecolor('#e74c3c')  # Red for losses
        elif left < 0:
            patch.set_facecolor('#f39c12')  # Orange for around zero
        else:
            patch.set_facecolor('#2ecc71')  # Green for gains

    # Add vertical lines for key metrics
    ax.axvline(x=stats['mean_return'], color='#2980b9', linewidth=2,
               linestyle='-', label=f"Mean: {stats['mean_return']:.1%}")
    ax.axvline(x=stats['median_return'], color='#27ae60', linewidth=2,
               linestyle='--', label=f"Median: {stats['median_return']:.1%}")
    ax.axvline(x=stats['var_95'], color='#c0392b', linewidth=2,
               linestyle=':', label=f"VaR 95%: {stats['var_95']:.1%}")
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    # Add percentile annotations
    percentiles = results['stress_scenario']['percentiles']

    ax.set_xlabel('1-Year Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"Distribution of Simulated Returns Under Stress ({stress['parameters']['n_simulations']} simulations)",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Add statistics box
    stats_text = (
        f"Std Dev: {stats['std_return']:.1%}\n"
        f"Min: {stats['min_return']:.1%}\n"
        f"Max: {stats['max_return']:.1%}\n"
        f"P(Loss > 10%): {stats['prob_loss_10pct']:.1%}\n"
        f"P(Loss > 20%): {stats['prob_loss_20pct']:.1%}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(results: Dict, output_path: str) -> None:
    """
    Create a 2x2 summary dashboard with all key visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Portfolio Weights (top-left)
    ax1 = axes[0, 0]
    weights = results['optimal_weights']
    weights_filtered = {k: v for k, v in weights.items() if v > 0.001}
    weights_sorted = dict(sorted(weights_filtered.items(), key=lambda x: -x[1]))

    tickers = list(weights_sorted.keys())
    values = list(weights_sorted.values())
    colors = ['#2ecc71' if t.endswith('.BK') else '#3498db' if t in ['LEMB', 'VWOB', 'EMLC']
              else '#9b59b6' if t == 'THB=X' else '#e74c3c' for t in tickers]

    ax1.bar(tickers, values, color=colors)
    ax1.set_ylabel('Weight')
    ax1.set_title('Optimal Portfolio Weights', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=8)

    # 2. Historical Backtest (top-right)
    ax2 = axes[0, 1]
    backtest = results['backtest_results']
    cumulative = backtest['cumulative_returns']
    dates = [parse_date(d) for d in cumulative.keys()]
    ax2.plot(dates, list(cumulative.values()), color='#2980b9', linewidth=1.5)
    ax2.fill_between(dates, 1, list(cumulative.values()), alpha=0.3, color='#3498db')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Portfolio Value')
    ax2.set_title(f"Historical Backtest (Return: {backtest['total_return']:.0%})", fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)

    # 3. Stress Simulation Paths (bottom-left)
    ax3 = axes[1, 0]
    stress = results['stress_scenario']
    paths = stress['simulation_paths']
    dates_stress = [parse_date(d) for d in paths['dates']]

    ax3.fill_between(dates_stress, paths['percentile_5'], paths['percentile_95'],
                     alpha=0.3, color='#e74c3c', label='5-95th pct')
    ax3.plot(dates_stress, paths['mean_path'], color='#c0392b', linewidth=2, label='Mean')
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_ylabel('Portfolio Value')
    ax3.set_title(f"Stress Scenario: {stress['name']}", fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Return Distribution (bottom-right)
    ax4 = axes[1, 1]
    final_returns = stress['all_simulations']['final_returns']
    stats = stress['statistics']

    ax4.hist(final_returns, bins=40, density=True, color='#3498db', edgecolor='white', alpha=0.7)
    ax4.axvline(x=stats['mean_return'], color='#2980b9', linewidth=2, label=f"Mean: {stats['mean_return']:.1%}")
    ax4.axvline(x=stats['var_95'], color='#c0392b', linewidth=2, linestyle=':', label=f"VaR 95%: {stats['var_95']:.1%}")
    ax4.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    ax4.set_xlabel('1-Year Return')
    ax4.set_ylabel('Density')
    ax4.set_title(f"Return Distribution (P(Loss): {stats['prob_loss']:.0%})", fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.suptitle('Portfolio Optimization & Stress Testing Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_all_visualizations(json_path: str, output_dir: str) -> None:
    """
    Generate all visualizations from results JSON.

    Args:
        json_path: Path to the results JSON file
        output_dir: Directory to save visualization files
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Load results
    print(f"\nLoading results from: {json_path}")
    results = load_results(json_path)

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualizations to: {output_dir}\n")

    # Generate all plots
    plot_optimal_weights(results, str(output_path / 'portfolio_weights.png'))
    plot_backtest_performance(results, str(output_path / 'backtest_performance.png'))
    plot_stress_simulation_paths(results, str(output_path / 'stress_simulation_paths.png'))
    plot_return_distribution(results, str(output_path / 'return_distribution.png'))
    plot_summary_dashboard(results, str(output_path / 'summary_dashboard.png'))

    print("\n" + "=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize Portfolio Optimization Results')
    parser.add_argument(
        '--input', '-i', type=str, default='results/portfolio_results.json',
        help='Input JSON results file (default: results/portfolio_results.json)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='results',
        help='Output directory for visualizations (default: results)'
    )

    args = parser.parse_args()
    create_all_visualizations(args.input, args.output_dir)


if __name__ == "__main__":
    main()
