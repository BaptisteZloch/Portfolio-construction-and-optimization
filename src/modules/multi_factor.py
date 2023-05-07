from typing import Literal
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import numpy.typing as npt


def reduce_dimentionality(
    full_dataframe: pd.DataFrame,
    mode: Literal["pca", "tsne"] = "pca",
    target_explained_variance: float = 0.95,
    result_as_df: bool = True,
) -> npt.NDArray[np.float64] | pd.DataFrame:
    """Reduce the dimension of a dataset given a target explained variance.

    Args:
    -----
        full_dataframe (pd.DataFrame): The dataframe to reduce the dimension.

        mode (Literal[&quot;pca&quot;, &quot;tsne&quot;], optional): The dimensionality reduction algorithm. Defaults to "pca".

        target_explained_variance (float, optional): The minimum threshold for explained variance. Defaults to 0.95.

        result_as_df (bool, optional): Whether or not return the result as a dataframe or as a numpy array. Defaults to True.

    Returns:
    -------
        npt.NDArray[np.float64] | pd.DataFrame: The reduced dataset.
    """
    assert (
        isinstance(full_dataframe, pd.DataFrame)
        and full_dataframe.shape[0] != 0
        and full_dataframe.shape[1] > 1
    ), "full_dataframe must be a pandas DataFrame containing at least 2 columns and 1 row"
    assert (
        target_explained_variance > 0 and target_explained_variance < 1
    ), "target_explained_variance must be a float between 0 and 1"
    X = full_dataframe.values
    match mode:
        case "pca":
            calibration_pca = PCA()
            calibration_pca.fit(X)

            d = (
                np.argmax(
                    np.cumsum(calibration_pca.explained_variance_ratio_)
                    >= target_explained_variance
                )
                + 1
            )
            pca = PCA(n_components=d)
            pca.fit(X)

            if result_as_df:
                return pd.DataFrame(
                    pca.transform(X), columns=[f"PC_{i}" for i in range(1, d + 1)]
                )
            return pca.transform(X)
        case "tsne":
            raise NotImplementedError("TSNE is not implemented yet")
        case _:
            raise ValueError("mode must be either pca or tsne")
