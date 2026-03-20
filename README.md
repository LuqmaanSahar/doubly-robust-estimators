# Doubly Robust Estimators Under Small Overlap
## D300_Coursework

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Coursework for the D300 module in Mphil Economics and Data Science

I run Monte Carlo simulations to investigate the finite sample performance of the doubly robust estimator when treatment effects are heterogeneous and overlap is poor. My approcah to simulations are based on: *Yang, Chengxin, Laine E. Thomas, and Fan Li. "Demystify Doubly-Robust Estimation: The Role of Overlap." arXiv preprint arXiv:2602.01648 (2026).*

The final report and figures can be found in the `report` folder.

The main analysis can be found in `orchestrator.ipynb`


## Summary
`overlapsim` is a Python package for running Monte Carlo simulations following the same
data generating process as (Yang et al., 2026), but altered to allow for a heterogeneous
treatment effect.

The model estimated is the `LinearDRLearner` from the `econML` package. It is an implementation
of the Augmented Inverse Probability Wieghts (AIPW) estimator.

The package allows for precise control over the data generating process, including the degree of
overlap and heterogeneity, the treatment prevalence, the correlation between features, and the
sample size.

---

## Package Structure

The `overlapsim` package follows the following simple structure:

- **data.py** – Module containing the data generator class.
- **metrics.py** – Helper functions for computing RMSE and bias.
- **simulation.py** – Module containg functions for running replications and full simulation runs.
- **plot_utils.py** – Module containing helpers for controlling plot styles.

And the main analyses can be found in:

- **orchestrator.ipynb** - Script for running all simulations and creating figures for report.

Additionally, the results from the simulations, and the raw figures can be found in the `results` folder.

The report itself along with final figures are in the `report` folder.

---

## Installation

### 1. Clone the repository

```bash
git clone # repository
cd D300_Coursework
```
### 2. Set up the Conda environment

```bash
conda env create -f environment.yml
conda activate overlapsim
```

### 3. Install the package locally

```bash
pip install -e .
```

---


## Usage

The following can also be found in `orchestrator.ipynb`.

Running simulations is simple:

```python
from overlapsim.data import SimulationData
from overlapsim.simulation import run_simulation
from overlapsim.metrics import summarise_results

# Initialise the simulation data class
gen = SimulationData(
    n_samples=500,
    overlap_coef=1,
    treatment_prevalence=0.1,
    heterogeneity_coef=0.5
)

# Replicate the simulation 1000 times.
# This returns a dataframe of the results from each replication
results = run_simulation(gen, n_replications=1000, base_seed=0)

# Create a summary of performance metrics for the full simulation run
summary = summarise_results(results)
```

## License

This project is licensed under the Apache-2.0 license. See LICENSE for details.