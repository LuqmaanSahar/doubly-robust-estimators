from __future__ import annotations

import numpy as np
from scipy.special import expit


class SimulationData:
    """
    Generate simulation data with a binary, heterogeneous treatment effect.
    Uses the same data generating process as Yang et al., 2026.

    The treatment effect is heterogeneous:
        tau(x) = 1 + heterogeneity_coef * x_1

    The treatment assignment follows a logistic propensity model:
        logit(e(x)) = intercept + overlap_coef * (0.2*X_1 + 0.3*X_2 + 0.4*X_3 - 0.25*X_4 - 0.3*X_5 - 0.3*X_6)

    And the outcome model given by:

        mu(x) = (0.2*X_1 + 0.3*X_2 + 0.4*X_3 - 0.25*X_4 - 0.3*X_5 - 0.3*X_6) + 

    Parameters
    ----------

    n_samples:
        Number of observations in sample
    n_features:
        Number of covariates/features.
    rho:
        Correlation coefficient between covariates.
    overlap_coef:
        Controls the amount of overlap in propensity scores.
        Higher values create less overlap.
    treatment_prevalence:
        The target treatment rate. Intercept is shifted to achieve this.
    heterogeneity_coef:
        Coefficient on X_1 in tau(x). Higher values indicate stronger
        heterogeneity.
    noise_std:
        Standard deviation of the random noise parameter. In all analyses, I
        keep this equal to 1 to match Yang et al., 2026.

    Returns
    -------
    dict
        Contains:
        - X: covariates
        - T: binary treatment
        - Y: observed outcome
        - propensity: true propensity scores
        - tau_true: true individual treatment effects
        - ate_true: true sample ATE

    """

    def __init__(
        self,
        n_samples: int = 500,
        n_features: int = 6,    # Keep at 6  to stay consistent with Yang et al., 2026
        rho: float = 0.5,   # Keep at 0.5 to stay consistent with Yang et al., 2026
        overlap_coef: float = 1.0,  # use 1 and 3
        treatment_prevalence: float = 0.4,   # use 0.1 and 0.4
        heterogeneity_coef: float = 0,
        noise_std: float = 1.0, # keep at 1 to stay consistent with Yang et al., 2026
    ) -> None:
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.rho = rho
        self.overlap_coef = overlap_coef
        self.treatment_prevalence = treatment_prevalence
        self.heterogeneity_coef = heterogeneity_coef
        self.noise_std = noise_std

    def _make_covariance(self) -> np.ndarray:
        """
        Create covariance matrix. I simulate data such that all covariates have
        the same correlation with one another, for simplicity.
        """
        cov = np.full((self.n_features, self.n_features), self.rho, dtype=float)
        np.fill_diagonal(cov, 1.0)
        return cov

    def _sample_features(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample 6 latent variables 

        Sample X1-X3 continuous and X4-X6 binary from latent Gaussian variables.
        """
        mean = np.zeros(self.n_features, dtype=float)
        cov = self._make_covariance()

        # Generate latent variables froma Gaussian distribution
        V = rng.multivariate_normal(mean=mean, cov=cov, size=self.n_samples)

        # Keep the first three features as Gaussian
        # Compute binary variables from the last three features
        X = V.copy()
        X[:, 3:6] = (V[:, 3:6] < 0).astype(float)

        return X

    def _propensity_linear(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear predictor without the intercept. Weights are chosen to 
        match Yang et al., 2026, to ensure meaningful comparisons.

        i.e. overlap_coef * (0.2*X_1 + 0.3*X_2 + 0.4*X_3 - 0.25*X_4 - 0.3*X_5 - 0.3*X_6)

        """
        beta = np.array([0.2, 0.3, 0.4, -0.25, -0.3, -0.3], dtype=float)
        return self.overlap_coef * (X[:, : len(beta)] @ beta)

    def _propensity_intercept(self, X: np.ndarray) -> float:
        """
        Shift the intercept of the treatment assignment model 
        to achieve the desired treatment prevalence.

        Choose the intercept that achieves a mean propensity score closest to target
        """
        linear_part = self._propensity_linear(X)
        grid = np.linspace(-6.0, 6.0, 2000)
        # compute the mean propensity score for each intercept in grid
        mean_props = np.array([expit(a + linear_part).mean() for a in grid])
        # Choose 'a' to get closest to target rate.
        idx = np.argmin(np.abs(mean_props - self.treatment_prevalence))
        return float(grid[idx])

    def _true_tau(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the true heterogeneous treatment effect tau(x).

        This is used to compute bias and the CATE-RMSE for a trained model.
        """
        return 1.0 + self.heterogeneity_coef * X[:, 0]

    def _baseline_outcome(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the baseline outcome function. Weights are chosen to match
        Yang et al. 2026, to ensure meaningful comparisons.

        Used to compute the actual outcomes for storing.
        """
        beta_y = np.array([-0.5, -0.8, -1.2, 0.8, 0.8, 1.0], dtype=float)
        return X[:, : len(beta_y)] @ beta_y

    def generate(self, random_state: int | None = None) -> dict[str, np.ndarray | float]:
        """
        Generate one simulated dataset.
        """
        rng = np.random.default_rng(random_state)

        X = self._sample_features(rng)
        intercept = self._propensity_intercept(X)

        linear_part = self._propensity_linear(X)
        propensity = expit(intercept + linear_part)
        T = rng.binomial(1, propensity).astype(float)

        tau_true = self._true_tau(X)
        mu = self._baseline_outcome(X)
        noise = rng.normal(0.0, self.noise_std, size=self.n_samples)

        Y = mu + T * tau_true + noise
        ate_true = float(np.mean(tau_true))

        return {
            "X": X,
            "T": T,
            "Y": Y,
            "propensity": propensity,
            "tau_true": tau_true,
            "ate_true": ate_true,
        }