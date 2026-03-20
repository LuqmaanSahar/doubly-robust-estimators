from pathlib import Path
from itertools import product
import pandas as pd

from overlapsim.data import SimulationData
from overlapsim.simulation import run_simulation
from overlapsim.metrics import summarise_results
from overlapsim.plot_utils import apply_gray_style


# Project root and results directory
project_root = Path.cwd()
results_dir = project_root / "results"


##############
# EXAMPLE: Running a single simulation with 1000 replications
##############

# Initialise the simulation data class
gen = SimulationData(
    n_samples=500,
    overlap_coef=1,
    treatment_prevalence=0.1,
    heterogeneity_coef=0.5
)

# Replicate the simulation 1000 times.
results = run_simulation(gen, n_replications=1000, base_seed=0)

# Create a summary of performance metrics for the full simulation run
summary = summarise_results(results)


##############
# Running all simulations
##############

heterogeneity_vals = [0, 0.5, 1, 2] # degree of heterogeneity
overlap_vals = [1, 3]   # degree of overlap
treatment_prev_vals = [0.1, 0.4]    # treatment prevalence
n_sample_vals = [300, 500, 1000, 2000]  # number of samples

all_results = []

# Run a simulation once for every combination in grid
for i, (heterogeneity_coef, overlap_coef, treatment_prevalence, n_samples) in enumerate(
    product(heterogeneity_vals, overlap_vals, treatment_prev_vals, n_sample_vals)
):
    # Display the simulation being run to keep track of where the loop is at
    print(
        f"[{i+1}/64] "
        f"heterogeneity={heterogeneity_coef}, "
        f"overlap={overlap_coef}, "
        f"treatment_prevalence={treatment_prevalence}, "
        f"n_samples={n_samples}"
    )

    # Initialise the data generating process with selected parameters
    gen = SimulationData(
        n_samples=n_samples,
        overlap_coef=overlap_coef,
        treatment_prevalence=treatment_prevalence,
        heterogeneity_coef=heterogeneity_coef,
    )

    # Replicate 1000 times per simulation
    results = run_simulation(
        generator=gen,
        n_replications=1000,
        base_seed=123 + 10000 * i,  # vary the base seed over simulations
    )

    # Append results over each loop to a single list
    all_results.append(results)

# Concatenate the results to a master dataset
full_results = pd.concat(all_results, ignore_index=True)

# Save the master dataset
full_results.to_csv(results_dir / "simulation_replication_results.csv", index=False)


# Apply the summary function to the dataset for each parameter combination
# Will create one row per parameter combination as a DataFrame
summary_table = (
    full_results
    .groupby(
        ["heterogeneity_coef", "overlap_coef", "treatment_prevalence", "n_samples"]
    )
    .apply(summarise_results)
    .reset_index()
)

# Save the summary table dataset
summary_table.to_csv(results_dir / "simulation_summary_results.csv", index=False)