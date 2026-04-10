%% Diebold-Li UK Yield Curve Workflow
% MATLAB workflow for the UK real forward curve application.
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

clearvars
close all
clc

%% Motivation
% Apply a Diebold-Li state-space model to UK real forward curve data and
% compute rolling one-step-ahead forecasts.

%% Project paths and dependency checks
scriptPath = string(mfilename('fullpath'));

% Running sections from the MATLAB editor can make mfilename unreliable for
% scripts, so fall back to the active editor file when needed.
if strlength(scriptPath) == 0 || ~isfile(scriptPath)
    if usejava('desktop')
        scriptPath = string(matlab.desktop.editor.getActiveFilename);
    end
end

if strlength(scriptPath) == 0 || ~isfile(scriptPath)
    error("run_diebold_li_uk:UnknownScriptPath", ...
        ['Could not determine the location of run_diebold_li_uk.m. ', ...
         'Open and run the file directly from matlab/scripts/.']);
end

scriptDirectory = fileparts(scriptPath);
matlabDirectory = fileparts(scriptDirectory);
functionsDirectory = fullfile(matlabDirectory, "functions");
projectRoot = fileparts(matlabDirectory);
inputDirectory = fullfile(projectRoot, "input", "glcrealddata");
outputDirectory = fullfile(projectRoot, "output");

if ~isfolder(functionsDirectory)
    error("run_diebold_li_uk:MissingFunctionsFolder", ...
        "Expected MATLAB helper folder not found: %s", functionsDirectory);
end

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
% Surface plot of the yield panel over time and maturity.

figure
surf(maturitiesYears, yieldCurve.dates, yieldCurve.yields)
datetick('y', 'yyyy')
ylabel('Date')
xlabel('Maturity (Years)')
zlabel('Forward Rate')
title('UK Real Forward Curve Surface')
colormap winter

%% Rolling one-step-ahead Diebold-Li forecasts
% For each rolling origin, estimate the model and produce a one-step-ahead
% forecast.

%% Forecast design
% Use "fast" for development and "full" for the denser rolling exercise.
runMode = "fast";

switch runMode
    case "fast"
        evaluationWindowDays = 60;
        forecastOriginStep = 5;
        selectedMaturitiesMonths = [48; 72; 120; 180; 240; 300];
        maxFunctionEvaluations = 5000;
        maxIterations = 200;
    case "full"
        evaluationWindowDays = 255;
        forecastOriginStep = 1;
        selectedMaturitiesMonths = maturitiesMonths;
        maxFunctionEvaluations = 25000;
        maxIterations = 1000;
    otherwise
        error("run_diebold_li_uk:UnknownRunMode", ...
            "runMode must be either ""fast"" or ""full"".");
end

selectedMaturityMask = ismember(maturitiesMonths, selectedMaturitiesMonths);
activeMaturitiesMonths = maturitiesMonths(selectedMaturityMask);
activeMaturitiesYears = maturitiesYears(selectedMaturityMask);
activeYieldCurve = yieldCurve;
activeYieldCurve.yields = yieldCurve.yields(:, selectedMaturityMask);

forecastHorizon = 1;
forecastOriginIndices = 1:forecastOriginStep:evaluationWindowDays;
numForecastOrigins = numel(forecastOriginIndices);
initialTrainingEnd = height(activeYieldCurve) - evaluationWindowDays;
numMaturities = numel(activeMaturitiesMonths);
MMSEForecasts = NaN(numForecastOrigins, numMaturities);
previousEstimatedParameters = [];

if evaluationWindowDays <= 0 || evaluationWindowDays >= height(activeYieldCurve)
    error("run_diebold_li_uk:InvalidForecastOrigins", ...
        "evaluationWindowDays must be positive and smaller than the sample length.");
end

if initialTrainingEnd + evaluationWindowDays > height(activeYieldCurve)
    error("run_diebold_li_uk:InvalidWindow", ...
        "The requested rolling window exceeds the available sample length.");
end

lambda0 = 0.0609;
X = [ones(size(activeMaturitiesMonths)), ...
    (1 - exp(-lambda0 * activeMaturitiesMonths)) ./ (lambda0 * activeMaturitiesMonths), ...
    ((1 - exp(-lambda0 * activeMaturitiesMonths)) ./ (lambda0 * activeMaturitiesMonths) - exp(-lambda0 * activeMaturitiesMonths))];

for originIndex = 1:numForecastOrigins
    forecastOrigin = forecastOriginIndices(originIndex);
    trainingEnd = initialTrainingEnd + forecastOrigin - 1;
    trainingSample = activeYieldCurve(1:trainingEnd, :);
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

    stateSpaceModel = ssm(@(params) Example_DieboldLi(params, yields, activeMaturitiesMonths));

    A0 = estimatedVarModel.AR{1};
    A0 = A0(:);
    Q0 = estimatedVarModel.Covariance;
    B0 = [sqrt(Q0(1,1)); 0; 0; sqrt(Q0(2,2)); 0; sqrt(Q0(3,3))];
    H0 = cov(residuals);
    D0 = sqrt(diag(H0));
    mu0 = mean(beta)';
    defaultInitialParameters = [A0; B0; D0; mu0; lambda0];

    if isempty(previousEstimatedParameters)
        initialParameters = defaultInitialParameters;
    else
        initialParameters = previousEstimatedParameters;
    end

    options = optimoptions('fminunc', ...
        'MaxFunEvals', maxFunctionEvaluations, ...
        'Algorithm', 'quasi-newton', ...
        'TolFun', 1e-8, ...
        'TolX', 1e-8, ...
        'MaxIter', maxIterations, ...
        'Display', 'off');

    [estimatedStateSpaceModel, parameters] = estimate( ...
        stateSpaceModel, yields, initialParameters, ...
        'Display', 'off', ...
        'Options', options, ...
        'Univariate', true, ...
        'Tolerance', 1e-15);

    mu = parameters(end-3:end-1)';
    intercept = estimatedStateSpaceModel.C * mu';
    deflatedYields = yields - intercept';

    [forecastedDeflatedYields, ~] = forecast(estimatedStateSpaceModel, ...
        forecastHorizon, deflatedYields);

    MMSEForecasts(originIndex, :) = forecastedDeflatedYields + intercept';
    previousEstimatedParameters = parameters;
end

%% Organize outputs
forecastDates = activeYieldCurve.dates(initialTrainingEnd + forecastOriginIndices);
forecastTable = array2table(MMSEForecasts, ...
    'VariableNames', matlab.lang.makeValidName("m" + string(activeMaturitiesMonths)));
forecastTable = addvars(forecastTable, forecastDates, 'Before', 1, 'NewVariableNames', "forecast_date");
forecastTable = addvars(forecastTable, ...
    forecastOriginIndices', ...
    'Before', 2, ...
    'NewVariableNames', {'forecast_origin_index'});

if runMode == "full"
    forecastMatrixFile = "MMSEForecasts.csv";
    forecastTableFile = "MMSEForecasts_table.csv";
else
    forecastMatrixFile = "MMSEForecasts_fast_ssm.csv";
    forecastTableFile = "MMSEForecasts_fast_ssm_table.csv";
end

outputMetadata = table( ...
    string(projectRoot), ...
    string(dependencyPath), ...
    string(runMode), ...
    numMaturities, ...
    join(string(activeMaturitiesMonths)', ","), ...
    evaluationWindowDays, ...
    forecastOriginStep, ...
    numForecastOrigins, ...
    initialTrainingEnd, ...
    forecastHorizon, ...
    'VariableNames', {'project_root', 'mathworks_dependency_path', 'run_mode', 'num_maturities_used', 'maturities_months_used', 'evaluation_window_days', 'forecast_origin_step', 'num_forecast_origins', 'initial_training_end', 'forecast_horizon'});

%% Save outputs
writematrix(MMSEForecasts, fullfile(outputDirectory, forecastMatrixFile));
writetable(forecastTable, fullfile(outputDirectory, forecastTableFile));
writetable(sampleSummary, fullfile(outputDirectory, "matlab_sample_summary.csv"));
writetable(outputMetadata, fullfile(outputDirectory, "matlab_run_metadata.csv"));

%% Illustrative forecast plot
% Plot one forecasted curve from the rolling exercise.

exampleForecastIndex = min(12, numForecastOrigins);

figure
plot(activeMaturitiesYears, MMSEForecasts(exampleForecastIndex, :)')
title(sprintf('One-step-ahead forecast at rolling origin %d', exampleForecastIndex))
xlabel('Maturity (Years)')
ylabel('Percent')
legend({'MMSE'}, 'Location', 'best')
