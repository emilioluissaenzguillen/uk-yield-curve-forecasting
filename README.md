# UK Yield Curve Forecasting: Academic Repository

This repository contains the empirical materials for a term-structure forecasting project developed in the context of Financial Econometrics at Bayes Business School. The project combines a state-space Diebold-Li specification with machine-learning components and supporting visualization scripts.

The repository is being reorganized from coursework-style working files into a more professional and reproducible academic structure. At this stage, the MATLAB workflow has been cleaned first because it provides the core term-structure benchmark and the clearest starting point for a replication-oriented repository.

## Research focus

The empirical application studies the dynamics and forecasting of the UK real yield curve. The current codebase includes:

- a MATLAB implementation of the Diebold-Li state-space model;
- Python code for forecast comparison and LSTM-based modeling;
- Python code for figure generation from saved outputs;
- archived scripts retained for traceability during the cleanup process.

## Repository structure

- `matlab/`: repository-facing MATLAB workflow, including a cleaned main script and helper functions.
- `python/`: repository-facing Python workspace for plotting, forecasting, and forecast comparison.
- `input/`: raw or externally sourced input data used by the empirical application.
- `output/`: generated tables, figures, and forecast files.
- `old/`: legacy or archival scripts kept temporarily during the restructuring process.

## MATLAB workflow

The main MATLAB entry point is:

- `matlab/scripts/run_diebold_li_uk.m`

This script is designed as the version-controlled source for a future MATLAB Live Script. It uses repository-relative paths and expects the MathWorks example support file `Example_DieboldLi.m` to be available locally through the MathWorks Diebold-Li example.

The original coursework-style MATLAB script has been archived as:

- `old/diebold_li_uk_legacy.m`

The MATLAB-specific notes are documented in:

- `matlab/README.md`

## Reproducibility note

The repository currently mixes cleaned files and legacy coursework files. This is intentional during the transition period so that results can still be traced back to the original working scripts.

At present:

- the MATLAB workflow has been partially standardized;
- the active Python scripts have been moved into the `python/` workspace, but the modeling code still has legacy structure;
- the legacy R plotting script has been archived and replaced by a Python plotting script;
- the original top-level MATLAB coursework script has been archived to keep the root focused on repository-facing entry points;
- no environment file or automated end-to-end pipeline has yet been added.

## Core references

The repository should cite both the academic model sources and the implementation source used for the MATLAB state-space workflow.

- Diebold, F.X., and C. Li (2006). "Forecasting the Term Structure of Government Bond Yields." Journal of Econometrics, 130(2), 337-364.
- Diebold, F.X., G.D. Rudebusch, and S.B. Aruoba (2006). "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach." Journal of Econometrics, 131, 309-338.
- Nelson, C.R., and A.F. Siegel (1987). "Parsimonious Modeling of Yield Curves." Journal of Business, 60(4), 473-489.
- MathWorks. "Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model." <https://www.mathworks.com/help/econ/using-the-kalman-filter-to-estimate-and-forecast-the-diebold-li-model.html>

## Citation

Repository-level citation metadata is provided in:

- `CITATION.cff`

Once this repository is published on GitHub, that file will allow GitHub to expose a machine-readable citation entry directly from the repository interface.

## License note

The repository code is released under the MIT License. See:

- `LICENSE`

This license applies to the code in the repository. External data sources, third-party materials, and MathWorks example dependencies remain subject to their own original terms.

## Current status

The repository is not yet fully publication-ready, but it now has a cleaner MATLAB foundation. The next steps are to:

- review and establish a clean commit history;
- decide which legacy files in `old/` are worth retaining long term;
- review whether additional repository metadata is needed beyond the current citation and license files;
- decide which generated outputs should be versioned and which should remain reproducible artifacts only.
