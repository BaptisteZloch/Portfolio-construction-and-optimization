from typing import Iterable, Literal, Optional
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from scipy.optimize import minimize


class ABCPortfolio(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_allocation(self):
        pass

    @staticmethod
    def _plot_allocation(
        weights: npt.NDArray | Iterable,
        assets: npt.NDArray | list[str] | tuple[str],
    ) -> None:
        """Plot the allocation of the portfolio in  a pie chart.

        Args:
            weights (npt.NDArray | Iterable): _description_
            assets (npt.NDArray | list[str] | tuple[str]): _description_
        """
        res = [(p, float(w)) for p, w in zip(assets, weights)]
        res.sort(key=lambda x: x[1], reverse=True)
        res = np.array(res)
        plt.title("Asset allocation")
        plt.pie(res[:, -1], labels=res[:, 0], autopct="%1.1f%%")
        plt.show()


class ConvexPortfolio(ABCPortfolio):
    def __init__(self, returns: pd.DataFrame) -> None:
        self.__returns = returns
        self.__already_optimized = False

    def fit(
        self,
        metric: Literal["sharpe", "risk", "return"] = "sharpe",
        way: Literal["min", "max"] = "max",
        max_asset_weight: float = 0.3,
        min_asset_weight: float = 0.0,
        trading_days: int = 365,
    ) -> None:
        """Optimize the portfolio using the Sharpe ratio, risk or return as a metric and a maximization or a minimization this metric. It uses an optimizer from `scipy.optimize` module. This method returns nothing.

        Args:
        -----
            metric (Literal[&quot;sharpe&quot;, &quot;risk&quot;, &quot;return&quot;], optional): The metric to optimize. Defaults to "sharpe".
            way (Literal[&quot;min&quot;, &quot;max&quot;], optional): The type of wanted optimization optimization. Defaults to "max".
            max_asset_weight (float, optional): The maximal weight of an asset in the portfolio. Defaults to 0.3.
            min_asset_weight (float, optional): The minimal weight of an asset in the portfolio. Defaults to 0.0.
            trading_days (int, optional): The number of trading days in a year, 365 for cryptos, 252 for stocks. Defaults to 365.

        """
        assert way in [
            "min",
            "max",
        ], "Invalid way of metric evaluation, must be a string equals to min or max."
        assert metric in [
            "sharpe",
            "risk",
            "return",
        ], "Invalid metric, must be a string equals to sharpe, risk or return."
        assert max_asset_weight <= 1.0, "Max asset weight must be less or equal to 1.0"
        assert (
            min_asset_weight >= 0.0
        ), "Min asset weight must be greater or equal to 0.0"
        assert (
            max_asset_weight >= min_asset_weight
        ), "Max asset weight must be greater or equal to min asset weight."

        def compute_metrics(
            weights: npt.NDArray,
            get_all_metrics: bool = False,
        ) -> float | dict[str, float | int]:  # Cost function
            weights = np.array(weights)
            ret = np.sum(self.__returns.mean() * weights * trading_days)
            vol = np.sqrt(weights.T @ self.__returns.cov() * trading_days @ weights)
            sr = ret / vol
            metrics = {"sharpe": sr, "risk": vol, "return": ret}
            if get_all_metrics:
                return metrics
            return (
                -metrics.get(metric, 0.0) if way == "max" else metrics.get(metric, 0.0)
            )

        cons = tuple(
            [
                {
                    "type": "eq",
                    "fun": lambda weights: np.sum(weights) - 1,
                },  # return 0 if sum of the weights is 1
            ]
        )

        bounds = tuple(
            [
                (min_asset_weight, max_asset_weight)
                for _ in range(len(self.__returns.columns))
            ]
        )
        init_guess = [
            1 / len(self.__returns.columns) for _ in range(len(self.__returns.columns))
        ]

        opt_results = minimize(
            compute_metrics, init_guess, method="SLSQP", bounds=bounds, constraints=cons
        )
        self.__already_optimized = True
        self.__optimized_weights = opt_results.x
        self.__optimized_metrics = compute_metrics(
            self.__optimized_weights, get_all_metrics=True
        )

    def get_allocation(self) -> pd.DataFrame:
        """Plot the allocation of the portfolio and return a DataFrame with the allocation of each asset. This function can't be called before the fit method.

        Returns:
        --------
            pd.DataFrame: The allocation of each asset in the portfolio, columns are the assets and rows are the weights.
        """
        assert (
            self.__already_optimized
        ), "You must fit the model before getting the allocation."

        print(f"{'  Results  ':-^40}")

        print(
            f"- Annualized Sharpe ratio: {self.__optimized_metrics.get('sharpe',0.0):.2f}\n- Annualized risk (volatility): {100*self.__optimized_metrics.get('risk',1.0):.2f} %\n- Annualized expected return: {100*self.__optimized_metrics.get('return',0.0):.2f} %"
        )
        ConvexPortfolio._plot_allocation(
            self.__optimized_weights, self.__returns.columns
        )
        return pd.DataFrame(
            {
                p: [w]
                for p, w in zip(
                    self.__returns.columns,
                    self.__optimized_weights,
                )
            }
        )


class MonteCarloPortfolio(ABCPortfolio):
    def __init__(self, returns: pd.DataFrame):
        self.__returns = returns
        self.__already_optimized = False

    def fit(
        self, n_portfolios: int = 20000, trading_days: int = 365, plot: bool = True
    ):
        """This method run a Monte Carlo simulation to allocate and create many portfolios in order to then find the most efficient. This method returns nothing.

        Args:
            n_portfolios (int, optional): The number of portefolio to create. Defaults to 20000.
            trading_days (int, optional): The number of trading days in a year, 365 for cryptos, 252 for stocks. Defaults to 365.
            plot (bool, optional): Whether or not to plot the portfolios simulated on a chart, x-axis = risk, y-axis = return. Defaults to True.
        """
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
        """Plot the portfolios simulated on a chart, x-axis = risk, y-axis = return.

        Args:
        -----
            n_portfolios (Optional[int], optional): The number of portfolio. Defaults to None.

        """
        assert (
            self.__already_optimized
        ), "You must fit the model before getting the allocation."
        plt.figure(figsize=(12, 8))
        plt.title(
            f"Monte Carlo Portfolio Optimization, ${n_portfolios}$ simulations"
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

    def get_allocation(
        self,
        metric: Literal["sharpe", "risk", "return"] = "sharpe",
        way: Literal["min", "max"] = "max",
    ) -> pd.DataFrame:
        """Plot the allocation of the portfolio and return a DataFrame with the allocation of each asset. This function can't be called before the fit method.

        Args:
            metric (Literal[&quot;sharpe&quot;, &quot;risk&quot;, &quot;return&quot;], optional): The metric to optimize. Defaults to "sharpe".
            way (Literal[&quot;min&quot;, &quot;max&quot;], optional): The type of wanted optimization optimization. Defaults to "max".

        Raises:
            Exception: _description_

        Returns:
            pd.DataFrame: _description_
        """
        assert (
            self.__already_optimized
        ), "You must fit the model before getting the allocation."
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
        MonteCarloPortfolio._plot_allocation(
            self.__all_weights[ind, :], self.__returns.columns
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
