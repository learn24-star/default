"""
Data Loader Module
Downloads and cleans financial data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def download_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 10
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD format). If None, calculated from years param
        end_date: End date (YYYY-MM-DD format). If None, uses today
        years: Number of years of historical data (default 10)

    Returns:
        DataFrame with adjusted close prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start = datetime.now() - timedelta(days=years * 365)
        start_date = start.strftime('%Y-%m-%d')

    print(f"Downloading data from {start_date} to {end_date}...")

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # Handle multi-level columns (yfinance returns multi-index for multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers

    return prices


def clean_data(
    prices: pd.DataFrame,
    max_na_ratio: float = 0.3,
    outlier_std: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Clean price data by removing NAs and outliers.

    Args:
        prices: DataFrame of prices
        max_na_ratio: Maximum ratio of NAs allowed per column (default 0.3)
        outlier_std: Number of standard deviations for outlier detection (default 5.0)

    Returns:
        Tuple of (cleaned prices, returns, list of removed tickers)
    """
    removed_tickers = []

    # Remove tickers with too many NAs
    na_ratios = prices.isna().sum() / len(prices)
    valid_tickers = na_ratios[na_ratios <= max_na_ratio].index.tolist()
    removed_by_na = na_ratios[na_ratios > max_na_ratio].index.tolist()
    removed_tickers.extend(removed_by_na)

    if removed_by_na:
        print(f"Removed tickers due to excessive NAs: {removed_by_na}")

    prices_clean = prices[valid_tickers].copy()

    # Forward fill then backward fill remaining NAs
    prices_clean = prices_clean.ffill().bfill()

    # Calculate returns
    returns = prices_clean.pct_change().dropna()

    # Remove outliers (replace with median)
    for col in returns.columns:
        col_mean = returns[col].mean()
        col_std = returns[col].std()
        lower_bound = col_mean - outlier_std * col_std
        upper_bound = col_mean + outlier_std * col_std

        outlier_mask = (returns[col] < lower_bound) | (returns[col] > upper_bound)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            print(f"Replaced {n_outliers} outliers in {col}")
            returns.loc[outlier_mask, col] = returns[col].median()

    # Align prices with returns
    prices_clean = prices_clean.loc[returns.index]

    return prices_clean, returns, removed_tickers


def get_clean_data(
    tickers: List[str],
    years: int = 10,
    max_na_ratio: float = 0.3,
    outlier_std: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Download and clean data in one step.

    Args:
        tickers: List of ticker symbols
        years: Number of years of historical data
        max_na_ratio: Maximum ratio of NAs allowed per column
        outlier_std: Number of standard deviations for outlier detection

    Returns:
        Tuple of (cleaned prices, returns, list of removed tickers)
    """
    prices = download_data(tickers, years=years)
    return clean_data(prices, max_na_ratio, outlier_std)


def calculate_statistics(returns: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for returns.

    Args:
        returns: DataFrame of returns

    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean_returns': returns.mean().to_dict(),
        'annual_returns': (returns.mean() * 252).to_dict(),
        'volatility': (returns.std() * np.sqrt(252)).to_dict(),
        'covariance_matrix': returns.cov().to_dict(),
        'correlation_matrix': returns.corr().to_dict()
    }
    return stats


if __name__ == "__main__":
    # Test the module
    tickers = [
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

    prices, returns, removed = get_clean_data(tickers, years=10)
    print(f"\nData shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Available tickers: {list(prices.columns)}")
    print(f"Removed tickers: {removed}")
