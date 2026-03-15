from __future__ import annotations

import numpy as np
import pandas as pd


def cate_rmse(tau_true: np.ndarray, tau_hat: np.ndarray) -> float:
    """
    Compute RMSE for individual treatment effect estimates.

    Parameters
    ----------
    tau_true:
        True individual treatment effects. (CATE)

    tau_hat:
        Model estimated treatment effects. (CATE)
    """
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def ate_bias(ate_true: float, ate_hat: float) -> float:
    """
    Compute bias of the overall estimated ATE.

    Parameters
    ----------
    ate_true:
        True sample ATE.
    ate_hat:
        Model estimated ATE for sample.
    """
    return float(ate_hat - ate_true)


def summarise_results(results: pd.DataFrame) -> pd.Series:
    """
    Aggregate replication-level simulation results.

    Expected columns:
    - ate_true
    - ate_hat
    - ate_bias
    - cate_rmse
    """
    return pd.Series(
        {
            "mean_ate_bias": results["ate_bias"].mean(),
            "rmse_ate": float(np.sqrt(np.mean(results["ate_bias"] ** 2))),
            "mean_cate_rmse": results["cate_rmse"].mean(),
            "sd_ate_hat": results["ate_hat"].std(ddof=1),
            "mean_treatment_rate": results["treatment_rate"].mean(),
        }
    )