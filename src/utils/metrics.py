"""
Financial Metrics Module
========================

This module provides functions for calculating financial performance metrics
used to evaluate portfolio strategies.

Why These Metrics?
------------------
Raw returns don't tell the whole story. A strategy that returns 20% but could
have lost 50% at any point is very different from one that returns 15% with
maximum 10% loss. These metrics capture different aspects of performance:

- **Return metrics**: How much money did we make?
- **Risk metrics**: How much could we have lost?
- **Risk-adjusted metrics**: How much return per unit of risk?

These are the metrics you'll report in your experiments and compare across
algorithms and replay strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def cumulative_return(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the total cumulative return.
    
    This answers: "If I started with £1, how much would I have at the end?"
    
    Parameters
    ----------
    returns : array-like
        Array of period returns (e.g., daily returns like 0.01 for 1% gain).
    
    Returns
    -------
    float
        Cumulative return (e.g., 0.5 means 50% total gain).
    
    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.03])  # +1%, -2%, +3%
    >>> cumulative_return(returns)
    0.0197...  # About 1.97% total
    
    Notes
    -----
    Formula: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    
    We multiply (1 + return) because:
    - If you have £100 and gain 10%, you have £100 * 1.10 = £110
    - If you then lose 5%, you have £110 * 0.95 = £104.50
    - Total return is £104.50 / £100 - 1 = 0.045 = 4.5%
    """
    returns = np.asarray(returns)
    return np.prod(1 + returns) - 1


def annualised_return(returns: Union[np.ndarray, pd.Series], 
                      periods_per_year: int = 252) -> float:
    """
    Calculate the annualised return.
    
    This standardises returns to a yearly basis for fair comparison.
    A 10% return over 2 years is different from 10% over 6 months.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    periods_per_year : int
        Number of periods in a year. Default 252 (trading days).
    
    Returns
    -------
    float
        Annualised return.
    
    Notes
    -----
    Formula: (1 + cumulative_return) ^ (periods_per_year / n_periods) - 1
    
    This is the geometric mean scaled to one year.
    """
    returns = np.asarray(returns)
    total_return = cumulative_return(returns)
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    # Handle negative total returns (can't raise negative to fractional power)
    if total_return <= -1:
        return -1.0
    
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def volatility(returns: Union[np.ndarray, pd.Series],
               annualise: bool = True,
               periods_per_year: int = 252) -> float:
    """
    Calculate the volatility (standard deviation of returns).
    
    Volatility measures how much returns vary. Higher volatility means
    more uncertainty—both more potential upside and downside.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    annualise : bool
        Whether to annualise the volatility.
    periods_per_year : int
        Number of periods in a year (for annualisation).
    
    Returns
    -------
    float
        Volatility (standard deviation).
    
    Notes
    -----
    Annualised volatility = daily volatility * sqrt(252)
    
    We use sqrt because volatility scales with the square root of time
    (this comes from the mathematics of random walks).
    """
    returns = np.asarray(returns)
    vol = np.std(returns, ddof=1)  # ddof=1 for sample standard deviation
    
    if annualise:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe Ratio.
    
    The Sharpe Ratio is the most important risk-adjusted metric. It answers:
    "How much excess return do I get per unit of risk?"
    
    A Sharpe of 1.0 means you get 1% extra return for each 1% of volatility.
    Higher is better. Generally:
    - < 0: Losing money
    - 0-1: Below average
    - 1-2: Good
    - 2-3: Very good
    - > 3: Exceptional (or possibly overfitting)
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    risk_free_rate : float
        Annualised risk-free rate (e.g., 0.02 for 2%).
    periods_per_year : int
        Number of periods in a year.
    
    Returns
    -------
    float
        Sharpe ratio.
    
    Notes
    -----
    Formula: (annualised_return - risk_free_rate) / annualised_volatility
    
    The risk-free rate is what you could earn with zero risk (e.g., Treasury bills).
    We subtract it because we only care about return *above* the risk-free rate.
    """
    returns = np.asarray(returns)
    
    ann_return = annualised_return(returns, periods_per_year)
    ann_vol = volatility(returns, annualise=True, periods_per_year=periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(returns: Union[np.ndarray, pd.Series],
                  risk_free_rate: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino Ratio.
    
    Similar to Sharpe, but only penalises *downside* volatility.
    The Sharpe ratio treats upside and downside volatility equally,
    but investors typically only dislike downside volatility.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    risk_free_rate : float
        Annualised risk-free rate.
    periods_per_year : int
        Number of periods in a year.
    
    Returns
    -------
    float
        Sortino ratio.
    
    Notes
    -----
    Formula: (annualised_return - risk_free_rate) / downside_deviation
    
    Downside deviation only considers returns below the risk-free rate.
    """
    returns = np.asarray(returns)
    
    ann_return = annualised_return(returns, periods_per_year)
    
    # Calculate daily risk-free rate for comparison
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Downside returns (only negative excess returns)
    downside_returns = returns[returns < daily_rf] - daily_rf
    
    if len(downside_returns) == 0:
        return np.inf  # No downside = infinite Sortino (be careful with this)
    
    downside_std = np.std(downside_returns, ddof=1)
    downside_std_annual = downside_std * np.sqrt(periods_per_year)
    
    if downside_std_annual == 0:
        return np.inf
    
    return (ann_return - risk_free_rate) / downside_std_annual


def max_drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the Maximum Drawdown (MDD).
    
    Maximum drawdown is the largest peak-to-trough decline. It answers:
    "What's the worst loss I would have experienced if I invested at the
    worst possible time and sold at the worst possible time?"
    
    This is crucial for understanding tail risk—even a strategy with great
    average returns might have a 50% drawdown that you couldn't stomach.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    
    Returns
    -------
    float
        Maximum drawdown as a positive number (e.g., 0.2 means 20% drawdown).
    
    Examples
    --------
    >>> returns = np.array([0.10, 0.05, -0.15, -0.10, 0.20])
    >>> max_drawdown(returns)
    0.2275  # About 22.75% maximum drawdown
    """
    returns = np.asarray(returns)
    
    # Calculate cumulative wealth (starting at 1)
    wealth = np.cumprod(1 + returns)
    
    # Running maximum wealth
    running_max = np.maximum.accumulate(wealth)
    
    # Drawdown at each point
    drawdowns = (running_max - wealth) / running_max
    
    return np.max(drawdowns)


def calmar_ratio(returns: Union[np.ndarray, pd.Series],
                 periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar Ratio.
    
    Calmar Ratio = Annualised Return / Maximum Drawdown
    
    This tells you how much return you get per unit of maximum drawdown.
    Unlike Sharpe (which uses volatility), Calmar focuses on the worst-case loss.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    periods_per_year : int
        Number of periods in a year.
    
    Returns
    -------
    float
        Calmar ratio.
    """
    returns = np.asarray(returns)
    
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    
    if mdd == 0:
        return np.inf if ann_ret > 0 else 0.0
    
    return ann_ret / mdd


def win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the win rate (percentage of profitable periods).
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    
    Returns
    -------
    float
        Win rate between 0 and 1.
    """
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        return 0.0
    
    return np.mean(returns > 0)


def profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the profit factor.
    
    Profit Factor = Gross Profit / Gross Loss
    
    A profit factor > 1 means you make more on winners than you lose on losers.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    
    Returns
    -------
    float
        Profit factor.
    """
    returns = np.asarray(returns)
    
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()  # Make positive
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def calculate_all_metrics(returns: Union[np.ndarray, pd.Series],
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> dict:
    """
    Calculate all performance metrics at once.
    
    This is a convenience function for evaluation and reporting.
    
    Parameters
    ----------
    returns : array-like
        Array of period returns.
    risk_free_rate : float
        Annualised risk-free rate.
    periods_per_year : int
        Number of periods in a year.
    
    Returns
    -------
    dict
        Dictionary containing all metrics.
    
    Examples
    --------
    >>> returns = np.random.randn(252) * 0.01 + 0.0005  # Simulated daily returns
    >>> metrics = calculate_all_metrics(returns)
    >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    returns = np.asarray(returns)
    
    return {
        'cumulative_return': cumulative_return(returns),
        'annualised_return': annualised_return(returns, periods_per_year),
        'volatility': volatility(returns, True, periods_per_year),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'n_periods': len(returns),
    }


# Make functions available at package level
__all__ = [
    'cumulative_return',
    'annualised_return',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'calculate_all_metrics',
]
