import os
from typing import Iterable, Optional, Tuple

import click
import pandas as pd
import pandas_datareader as pdr
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return, ema_historical_return
from pypfopt.risk_models import sample_cov, semicovariance, exp_cov, min_cov_determinant


EXPECTED_RETURN_METHODOLOGY = {"mean": mean_historical_return, "ema": ema_historical_return}
RISK_MODELS = {
    "sample": sample_cov,
    "semi": semicovariance,
    "exp": exp_cov,
    "min_det": min_cov_determinant,
}


def get_stock_data(tickers: Iterable[str], tiingo_api_key: str) -> pd.DataFrame:
    asset_data = pdr.get_data_tiingo(symbols=tickers, api_key=tiingo_api_key).reset_index()
    # format the price data in the manner that PyPortfolioOpt expects it which is where the columns
    # are the tickers and their respective prices indexed by date. Use adjClose to account for
    # splits, dividends, etc.
    formatted_df = asset_data.pivot(index="date", columns="symbol", values="adjClose")
    return formatted_df


@click.command()
@click.argument("tickers", nargs=-1)
@click.option(
    "--strategy",
    "-s",
    default="max_sharpe",
    type=click.Choice(["max_sharpe", "min_vol", "eff_risk", "eff_return"], case_sensitive=False),
)
@click.option(
    "--expected_return",
    "-e",
    default="mean",
    type=click.Choice(EXPECTED_RETURN_METHODOLOGY.keys(), case_sensitive=False),
)
@click.option(
    "--risk_model",
    "-r",
    default="sample",
    type=click.Choice(RISK_MODELS.keys(), case_sensitive=False),
)
@click.option("--portfolio_value", "-p", default=100000, type=float)
@click.option("--risk_free", type=float, default=0.03)
@click.option("--risk_aversion", type=float, default=1)
@click.option("--target_risk", type=float)
@click.option("--target_return", type=float)
@click.option("--tiingo_api_key")
@click.option("--verbose/--no-verbose", "-v/ ", default=False)
def calculate_optimized_portfolio(
    tickers: Tuple[str],
    strategy: str,
    expected_return: str,
    risk_model: str,
    portfolio_value: float,
    risk_free: float,
    risk_aversion: float,
    target_risk: float,
    target_return: float,
    tiingo_api_key: Optional[str],
    verbose: bool,
) -> None:

    tiingo_api_key = tiingo_api_key or os.environ.get("TIINGO_API_KEY")
    if tiingo_api_key is None:
        raise RuntimeError(
            "Tiingo API key not found. Please pass in an api key or set the "
            "TIINGO_API_KEY environment variable"
        )
    stock_data = get_stock_data(tickers, tiingo_api_key)
    expected_returns = EXPECTED_RETURN_METHODOLOGY[expected_return](stock_data)
    cov_matrix = RISK_MODELS[risk_model](stock_data)
    efficient_frontier = EfficientFrontier(expected_returns, cov_matrix)

    if strategy == "max_sharpe":
        raw_weights = efficient_frontier.max_sharpe(risk_free_rate=risk_free)
    elif strategy == "min_vol":
        raw_weights = efficient_frontier.min_volatility()
    elif strategy == "eff_risk":
        raw_weights = efficient_frontier.efficient_risk(
            target_risk=target_risk, risk_free_rate=risk_free
        )
    elif strategy == "eff_return":
        raw_weights = efficient_frontier.efficient_return(target_return=target_return)

    cleaned_weights = efficient_frontier.clean_weights()
    click.echo(cleaned_weights)
    latest_prices = get_latest_prices(stock_data)
    discrete_weights = DiscreteAllocation(cleaned_weights, latest_prices, portfolio_value)
    discrete_weights.lp_portfolio(verbose=verbose)
    efficient_frontier.portfolio_performance(verbose=True)


if __name__ == "__main__":
    calculate_optimized_portfolio()
