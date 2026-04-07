# MATLAB Workflow

This folder contains the repository-facing MATLAB implementation of the UK real yield curve application based on the Diebold-Li framework.

## Recommended convention

For an academic GitHub repository, the best compromise is usually:

- keep a plain-text `.m` script as the canonical source file for version control and code review;
- generate a `.mlx` Live Script from that `.m` file for presentation, explanation, and classroom use.

This approach keeps the project readable on GitHub while still giving you the richer notebook-style MATLAB experience.

## Current structure

- `scripts/run_diebold_li_uk.m`: main entry script with section breaks suitable for conversion to a Live Script.
- `functions/load_glc_real_curve_data.m`: reusable loader for the Bank of England GLC real forward curve spreadsheets.
- `functions/check_mathworks_example_dependency.m`: helper that validates the external MathWorks example dependency.

## How to create the Live Script

1. Open `scripts/run_diebold_li_uk.m` in MATLAB.
2. Review or expand the section text and outputs.
3. Use `Save As` and choose `MATLAB Live Script (*.mlx)`.
4. Save the converted file alongside the script, for example as `scripts/run_diebold_li_uk.mlx`.

The script is already structured around notebook-style sections so that the resulting Live Script reads more like an academic replication document than a raw code file.

## Reproducibility note

The state-space helper `Example_DieboldLi.m` is referenced by the original coursework code but is not currently present in this repository. In MATLAB, it can appear to be "already there" because MathWorks examples can download supporting files into your local `Examples` area and open them for you.

According to the MathWorks documentation for `openExample`, MATLAB examples and their supporting files are downloaded locally when you open them, which is why `Example_DieboldLi.m` can feel built-in even though it is better understood as an example support file.

For this repository, the intended approach is to depend on the MathWorks example support file rather than store a copy in the repo. A reproducible setup can therefore instruct users to obtain the support file from the MathWorks example and confirm its location with `which Example_DieboldLi -all`.

The new script therefore checks for that dependency explicitly and stops with a clear error message if it is missing.

## Dependency setup

Before running the repository script, open the MathWorks Diebold-Li example once from the MATLAB Help Center or from the MathWorks documentation page so that the supporting files are available locally.

Recommended check inside MATLAB:

```matlab
which Example_DieboldLi -all
```

If `which` returns a valid path, the repository script should be able to use the example helper.

## Output files

When the script runs successfully, it is designed to produce:

- `output/MMSEForecasts.csv`: raw forecast matrix.
- `output/MMSEForecasts_table.csv`: forecast matrix with forecast dates and maturity labels.
- `output/matlab_sample_summary.csv`: sample coverage summary for the MATLAB workflow.
- `output/matlab_run_metadata.csv`: lightweight run metadata, including the resolved MathWorks dependency path.

## Citation and attribution

This repository should cite both the academic sources for the model and the implementation source that inspired the MATLAB state-space workflow.

- Primary methodology: Diebold and Li (2006), "Forecasting the Term Structure of Government Bond Yields."
- State-space extension and macro interpretation: Diebold, Rudebusch, and Aruoba (2006), "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach."
- MATLAB implementation reference: MathWorks, "Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model," available at <https://www.mathworks.com/help/econ/using-the-kalman-filter-to-estimate-and-forecast-the-diebold-li-model.html>.

In an academic-style GitHub repository, the MathWorks page is best presented as an implementation and software reference, while the journal articles remain the primary scholarly citations.

## Suggested next repository step

After the missing helper function is restored, the next cleanup pass should separate:

- raw inputs in `input/`,
- generated tables in `output/`,
- archival code in `old/`,
- and language-specific workflows for MATLAB, Python, and R.
