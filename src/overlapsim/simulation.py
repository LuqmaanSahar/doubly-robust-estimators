from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dr import LinearDRLearner
from sklearn.linear_model import LinearRegression, LogisticRegression

from overlapsim.metrics import ate_bias, cate_rmse


def correct_drlearner(random_state: int | None = None) -> LinearDRLearner:
    """
    Build a correctly specified LinearDRLearner.
    """
    return LinearDRLearner(
        model_propensity=LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        model_regression=LinearRegression(),
        cv=3,
        random_state=random_state,
    )


def run_one(generator, random_state: int | None = None) -> dict:
    """
    Run one full replication:
    1. simulate a dataset,
    2. fit LinearDRLearner,
    3. compute key performance metrics.


    Parameters
    ----------

    generator:
        The simulated data generator class
    random_state:


    Returns:
    --------
    dict
        Contains:
        - ate_true: Ground truth ATE
        - ate_hat: Model estimated ATE in sample
        - ate_bias: Difference between actual and estimated ATE
        - cate_rmse: RMSE for the estimated CATEs
        - treatment_rate: The actual treated population in random sample
        - min_propensity: The minimum observed propensity score
        - max_propensity: The maximum observed propensity score
    """
    data = generator.generate(random_state=random_state)

    X = data["X"]
    T = data["T"]
    Y = data["Y"]
    tau_true = data["tau_true"]
    ate_true = data["ate_true"]

    # Train the model
    model = correct_drlearner(random_state=random_state)
    model.fit(Y, T, X=X)

    # Recover the estimated CATE for each observation in sample
    tau_hat = np.asarray(model.effect(X), dtype=float).reshape(-1)
    # Take the mean to recover the estimated ATE
    ate_hat = float(np.mean(tau_hat))

    return {
        "ate_true": ate_true,
        "ate_hat": ate_hat,
        "ate_bias": ate_bias(ate_true, ate_hat),
        "cate_rmse": cate_rmse(tau_true, tau_hat),
        "treatment_rate": float(np.mean(T))
    }


def run_simulation(generator, n_replications: int, base_seed: int = 123) -> pd.DataFrame:
    """
    Run many replications under a fixed DGP using a Monte Carlo setup.

    Parameters:
    -----------
    generator:
        The simulated data generator class.
    n_replications:
        Number of times to replicate the simulation on a new seed.
    base_seed:
        Seed number to use as first random state.
    """
    rows = []

    for rep in range(n_replications):
        # Call run_one and change the seed every replication
        result = run_one(generator, random_state=base_seed + rep)

        # Save metadata for clean bookkeeping
        # Important for sanity checks once we save the results
        result["replication"] = rep
        result["n_samples"] = generator.n_samples
        result["overlap_coef"] = generator.overlap_coef
        result["treatment_prevalence"] = generator.treatment_prevalence
        result["heterogeneity_coef"] = generator.heterogeneity_coef
        rows.append(result)

    # convert list of dictionaries to a dataframe and return
    return pd.DataFrame(rows)