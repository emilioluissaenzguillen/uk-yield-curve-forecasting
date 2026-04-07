%% Diebold-Li UK Yield Curve Workflow
% Academic replication workflow for the UK real forward curve application.
% This script is designed as the plain-text source for a MATLAB Live Script
% and is intended to support a professional GitHub repository.
%
% Implementation note:
%   The state-space estimation workflow and the missing helper function
%   Example_DieboldLi.m are closely aligned with the MathWorks tutorial
%   "Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model":
%   https://www.mathworks.com/help/econ/using-the-kalman-filter-to-estimate-and-forecast-the-diebold-li-model.html
%
% Primary academic references:
%   Diebold, F.X., and C. Li (2006), "Forecasting the Term Structure of
%   Government Bond Yields", Journal of Econometrics, 130(2), 337-364.
%   Diebold, F.X., G.D. Rudebusch, and S.B. Aruoba (2006),
%   "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor
%   Approach", Journal of Econometrics, 131, 309-338.
%
% Dependencies:
%   - Econometrics Toolbox
%   - Statistics and Machine Learning Toolbox
%   - MathWorks example support file Example_DieboldLi.m
%
% Suggested Live Script sections:
%   1. Motivation and references
%   2. Data description
%   3. Yield curve visualization
%   4. Rolling Diebold-Li estimation
%   5. Forecast output and interpretation

clearvars
close all
clc

%% Motivation
% This workflow applies a Diebold-Li term-structure model in state-space
% form to UK real forward curve data. The goal is to estimate latent level,
% slope, and curvature factors and then compute rolling one-step-ahead MMSE
% forecasts across the maturity spectrum.
%
% The empirical design follows the structure of the MathWorks tutorial for
% state-space estimation, adapted here to UK data and to the repository's
% directory layout.

%% Project paths and dependency checks
scriptDirectory = fileparts(mfilename('fullpath'));
matlabDirectory = fileparts(scriptDirectory);
functionsDirectory = fullfile(matlabDirectory, "functions");
projectRoot = fileparts(matlabDirectory);
inputDirectory = fullfile(projectRoot, "input", "glcrealddata");
outputDirectory = fullfile(projectRoot, "output");

addpath(functionsDirectory);

dependencyPath = check_mathworks_example_dependency();

if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end

%% Load data
dataset = load_glc_real_curve_data(inputDirectory);
yieldCurve = dataset.yield_curve;
maturitiesMonths = dataset.maturities_months;
maturitiesYears = dataset.maturities_years;

sampleSummary = table( ...
    min(yieldCurve.dates), ...
    max(yieldCurve.dates), ...
    height(yieldCurve), ...
    numel(maturitiesMonths), ...
    'VariableNames', {'sample_start', 'sample_end', 'num_observations', 'num_maturities'});

disp("Sample summary")
disp(sampleSummary)
disp("MathWorks dependency in use")
disp(string(dependencyPath))

%% Visualize the yield curve panel
% The surface plot provides a compact view of how the cross-section of real
% forward rates evolves over time.

figure
surf(maturitiesYears, yieldCurve.dates, yieldCurve.yields)
datetick('y', 'yyyy')
ylabel('Date')
xlabel('Maturity (Years)')
zlabel('Forward Rate')
title('UK Real Forward Curve Surface')
colormap winter

%% Rolling one-step-ahead Diebold-Li forecasts
% The specification below mirrors the standard three-factor Diebold-Li
% framework. For each rolling origin, the script:
%   1. estimates latent factors by OLS with fixed lambda;
%   2. fits a VAR(1) to the factor series for initial values;
%   3. estimates the state-space model by maximum likelihood;
%   4. produces a one-step-ahead MMSE forecast for the yield curve.

forecastOrigins = 255;
initialTrainingEnd = 5438;
forecastHorizon = 1;
numMaturities = numel(maturitiesMonths);
MMSEForecasts = NaN(forecastOrigins, numMaturities);

if initialTrainingEnd + forecastOrigins - 1 > height(yieldCurve)
    error("run_diebold_li_uk:InvalidWindow", ...
        "The requested rolling window exceeds the available sample length.");
end

lambda0 = 0.0609;
X = [ones(size(maturitiesMonths)), ...
    (1 - exp(-lambda0 * maturitiesMonths)) ./ (lambda0 * maturitiesMonths), ...
    ((1 - exp(-lambda0 * maturitiesMonths)) ./ (lambda0 * maturitiesMonths) - exp(-lambda0 * maturitiesMonths))];

for forecastOrigin = 1:forecastOrigins
    trainingEnd = initialTrainingEnd + forecastOrigin - 1;
    trainingSample = yieldCurve(1:trainingEnd, :);
    yields = trainingSample.yields;

    beta = zeros(size(yields, 1), 3);
    residuals = zeros(size(yields, 1), numMaturities);

    for observationIndex = 1:size(yields, 1)
        olsModel = fitlm(X, yields(observationIndex, :)', 'Intercept', false);
        beta(observationIndex, :) = olsModel.Coefficients.Estimate';
        residuals(observationIndex, :) = olsModel.Residuals.Raw';
    end

    varModel = varm(3, 1);
    estimatedVarModel = estimate(varModel, beta);

    stateSpaceModel = ssm(@(params) Example_DieboldLi(params, yields, maturitiesMonths));

    A0 = estimatedVarModel.AR{1};
    A0 = A0(:);
    Q0 = estimatedVarModel.Covariance;
    B0 = [sqrt(Q0(1,1)); 0; 0; sqrt(Q0(2,2)); 0; sqrt(Q0(3,3))];
    H0 = cov(residuals);
    D0 = sqrt(diag(H0));
    mu0 = mean(beta)';
    initialParameters = [A0; B0; D0; mu0; lambda0];

    options = optimoptions('fminunc', ...
        'MaxFunEvals', 25000, ...
        'Algorithm', 'quasi-newton', ...
        'TolFun', 1e-8, ...
        'TolX', 1e-8, ...
        'MaxIter', 1000, ...
        'Display', 'off');

    [estimatedStateSpaceModel, parameters] = estimate(stateSpaceModel, yields, ...
        initialParameters, 'Display', 'off', 'Options', options, 'Univariate', true);

    mu = parameters(end-3:end-1)';
    intercept = estimatedStateSpaceModel.C * mu';
    deflatedYields = yields - intercept';

    [forecastedDeflatedYields, ~] = forecast(estimatedStateSpaceModel, ...
        forecastHorizon, deflatedYields);

    MMSEForecasts(forecastOrigin, :) = forecastedDeflatedYields + intercept';
end

%% Organize outputs
forecastDates = yieldCurve.dates(initialTrainingEnd + 1:initialTrainingEnd + forecastOrigins);
forecastTable = array2table(MMSEForecasts, ...
    'VariableNames', matlab.lang.makeValidName("m" + string(maturitiesMonths)));
forecastTable = addvars(forecastTable, forecastDates, 'Before', 1, 'NewVariableNames', "forecast_date");

outputMetadata = table( ...
    string(projectRoot), ...
    string(dependencyPath), ...
    forecastOrigins, ...
    initialTrainingEnd, ...
    forecastHorizon, ...
    'VariableNames', {'project_root', 'mathworks_dependency_path', 'forecast_origins', 'initial_training_end', 'forecast_horizon'});

%% Save outputs
writematrix(MMSEForecasts, fullfile(outputDirectory, "MMSEForecasts.csv"));
writetable(forecastTable, fullfile(outputDirectory, "MMSEForecasts_table.csv"));
writetable(sampleSummary, fullfile(outputDirectory, "matlab_sample_summary.csv"));
writetable(outputMetadata, fullfile(outputDirectory, "matlab_run_metadata.csv"));

%% Illustrative forecast plot
% The plot below highlights one forecasted curve from the rolling exercise.

exampleForecastIndex = 12;

figure
plot(maturitiesYears, MMSEForecasts(exampleForecastIndex, :)')
title(sprintf('One-step-ahead forecast at rolling origin %d', exampleForecastIndex))
xlabel('Maturity (Years)')
ylabel('Percent')
legend({'MMSE'}, 'Location', 'best')

%% Interpretation notes
% In the Live Script version, this section is a good place to discuss:
%   - whether the forecasted curve preserves plausible shape restrictions;
%   - how the UK application compares with the U.S. tutorial benchmark;
%   - which maturities appear hardest to fit or forecast;
%   - how the state-space estimates differ from simpler two-step estimates.
