from typing import Literal, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MonteCarloPortfolio:
    def __init__(self, returns: pd.DataFrame):
        self.__returns = returns
        self.__already_optimized = False

    def fit(
        self, n_portfolios: int = 20000, trading_days: int = 365, plot: bool = True
    ):
        self.__all_weights = np.zeros((n_portfolios, len(self.__returns.columns)))
        self.__ret_arr = np.zeros(n_portfolios)
        self.__vol_arr = np.zeros(n_portfolios)
        self.__sharpe_arr = np.zeros(n_portfolios)

        cov_matrix = self.__returns.cov()
        rets = self.__returns.mean()

        for x in tqdm(range(n_portfolios)):
            # Weights
            weights = np.array(
                np.random.uniform(size=len(self.__returns.columns)), dtype=np.float64
            )
            weights = weights / np.sum(weights)

            # Save weights
            self.__all_weights[x, :] = weights

            # Expected return
            self.__ret_arr[x] = np.sum((rets * weights * trading_days))

            # Expected volatility
            self.__vol_arr[x] = np.sqrt(weights.T @ cov_matrix * trading_days @ weights)

            # Sharpe Ratio
            self.__sharpe_arr[x] = self.__ret_arr[x] / self.__vol_arr[x]
        self.__already_optimized = True

        if plot:
            self.plot_portfolios(n_portfolios)

    def plot_portfolios(self, n_portfolios: Optional[int] = None) -> None:
        if not self.__already_optimized:
            raise Exception("You need to fit first.")
        plt.figure(figsize=(12, 8))
        plt.title(
            f"Monte Carlo Portfolio Optimization, {n_portfolios} simulations"
            if n_portfolios
            else "Monte Carlo Portfolio Optimization"
        )
        plt.grid(True)
        plt.scatter(self.__vol_arr, self.__ret_arr, c=self.__sharpe_arr, cmap="viridis")
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.scatter(
            self.__vol_arr[self.__sharpe_arr.argmax()],
            self.__ret_arr[self.__sharpe_arr.argmax()],
            c="red",
            s=50,
        )  # red dot
        plt.show()

    @staticmethod
    def __plot_allocation(weights, assets):
        plt.title("Asset allocation")
        plt.pie(weights, labels=assets, autopct="%1.1f%%")
        plt.show()

    def get_allocation(
        self,
        metric: Literal["sharpe", "risk", "return"] = "sharpe",
        way: Literal["min", "max"] = "max",
    ) -> pd.DataFrame:
        if not self.__already_optimized:
            raise Exception("You need to fit first.")
        assert way in [
            "min",
            "max",
        ], "Invalid way of metric evaluation, must be a string equals to min or max."

        match metric:
            case "sharpe":
                ind = (
                    self.__sharpe_arr.argmin()
                    if way == "min"
                    else self.__sharpe_arr.argmax()
                )
            case "risk":
                ind = (
                    self.__vol_arr.argmin() if way == "min" else self.__vol_arr.argmax()
                )
            case "return":
                ind = (
                    self.__vol_arr.argmin() if way == "min" else self.__vol_arr.argmax()
                )
            case _:
                raise Exception("Invalid metric.")

        print(f"{'  Results  ':-^40}")
        print(
            f"- Annualized Sharpe ratio: {self.__sharpe_arr[ind]:.2f}\n- Annualized risk (volatility): {100*self.__vol_arr[ind]:.2f} %\n- Annualized expected return: {100*self.__ret_arr[ind]:.2f} %"
        )
        weighted_assets = [
            (p, w)
            for p, w in zip(
                self.__returns.columns,
                self.__all_weights[ind, :],
            )
        ]
        weighted_assets.sort(key=lambda x: x[1], reverse=True)
        weighted_assets = np.array(weighted_assets)
        MonteCarloPortfolio.__plot_allocation(
            weighted_assets[:, -1], weighted_assets[:, 0]
        )
        return pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self.__returns.columns,
                    self.__all_weights[ind, :],
                )
            }
        )
