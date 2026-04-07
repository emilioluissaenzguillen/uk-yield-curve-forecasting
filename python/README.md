# Python Workflow

This folder is intended for the repository-facing Python components of the project.

## Planned scope

- LSTM-based yield curve forecasting
- forecast comparison utilities such as the Diebold-Mariano test
- figure generation from repository outputs

## Current scripts

- `scripts/lstm_uk.py`: legacy LSTM forecasting workflow, now adapted to use repository-relative paths.
- `scripts/dm_test.py`: Diebold-Mariano test utility used by the Python workflow.
- `scripts/generate_figures.py`: figure-generation script replacing the legacy R plotting workflow.

## Plotting workflow

The script `scripts/generate_figures.py` replaces the legacy R plotting workflow and reproduces the main comparison figures directly from the CSV files stored in `output/`.

It is designed to generate:

- `output/4y.png`
- `output/15y.png`
- `output/25y.png`
- `output/equiv_rate.png`
- `output/4y_dieboldli.png`
- `output/15y_dieboldli.png`
- `output/25y_dieboldli.png`
- `output/equiv_rate_dbli.png`
- `output/diebold_li_panel.png`

## Typical usage

From the repository root:

```bash
python python/scripts/generate_figures.py
```

The script assumes that the underlying CSV outputs already exist in `output/`.

The LSTM workflow can likewise be run from the repository root with:

```bash
python python/scripts/lstm_uk.py
```
