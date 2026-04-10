# UK Yield Curve Forecasting

This repository contains code and data workflows for forecasting the UK yield curve. It combines a state-space Diebold-Li benchmark in MATLAB with Python-based machine-learning, forecast comparison, and figure-generation scripts.

## Research focus

The empirical application studies the dynamics and forecasting of the UK real yield curve. The current repository includes:

- a MATLAB implementation of the Diebold-Li state-space model;
- Python code for forecast comparison and LSTM-based modeling;
- Python code for figure generation from saved outputs;
- repository metadata for citation and licensing.

## Repository structure

- `matlab/`: repository-facing MATLAB workflow, including a cleaned main script and helper functions.
- `python/`: repository-facing Python workspace for plotting, forecasting, and forecast comparison.
- `input/`: raw or externally sourced input data used by the empirical application.
- `output/`: generated tables, figures, and forecast files.

## MATLAB workflow

The main MATLAB entry point is:

- `matlab/scripts/run_diebold_li_uk.m`

This script is designed as the version-controlled source for the MATLAB workflow. It uses repository-relative paths and expects the MathWorks example support file `Example_DieboldLi.m` to be available locally through the MathWorks Diebold-Li example.

The MATLAB-specific notes are documented in:

- `matlab/README.md`

## Reproducibility note

The repository has been simplified to focus on the active MATLAB and Python workflows. It does not yet include a full environment specification or a fully automated end-to-end pipeline, so users should treat it as a research codebase rather than a one-command replication package.

## Core references

The repository should cite both the academic model sources and the implementation source used for the MATLAB state-space workflow:

- Diebold, F.X., and C. Li (2006). "Forecasting the Term Structure of Government Bond Yields." Journal of Econometrics, 130(2), 337-364.
- Diebold, F.X., G.D. Rudebusch, and S.B. Aruoba (2006). "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach." Journal of Econometrics, 131, 309-338.
- Nelson, C.R., and A.F. Siegel (1987). "Parsimonious Modeling of Yield Curves." Journal of Business, 60(4), 473-489.
- MathWorks. "Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model." <https://www.mathworks.com/help/econ/using-the-kalman-filter-to-estimate-and-forecast-the-diebold-li-model.html>

## Citation

Repository-level citation metadata is provided in:

- `CITATION.cff`

## License note

The repository code is released under the MIT License. See:

- `LICENSE`

This license applies to the code in the repository. External data sources, third-party materials, and MathWorks example dependencies remain subject to their own original terms.

## Current status

The repository now has a clean public structure. The main remaining improvements are to:

- create the MATLAB `.mlx` version directly from within MATLAB;
- decide which generated outputs should remain versioned in `output/`;
- optionally add an environment file for the Python workflow;
- review whether additional repository metadata is needed beyond the current citation and license files;
