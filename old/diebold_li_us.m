%% Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model 
% SMM943 Financial Econometrics - Coursework      
% Author: Emilio Luis Sáenz Guillén, PhD Student in Actuarial Science  
% Professor: Giovanni Urga                                         
% Bayes Business School;December 2022 

clc
clear
cd 'C:\Users\addj700\Dropbox\Bayes Business School\PhD\Modules\1st Term\Financial Econometrics (SMM943)\Presentation - ML\Simulation'

url = 'https://fred.stlouisfed.org/';
link = fred(url);

format bank

startdate = '1993-10-01'; % beginning of date range for historical data
enddate = '2022-07-15'; % ending of date range for historical data

series = {"DGS3MO","DGS6MO","DGS1","DGS2","DGS3","DGS5","DGS7","DGS10","DGS20","DGS30"};

dates = datetime(fetch(link,series{1},startdate,enddate).Data(:,1),'ConvertFrom','datenum');

NameObs = strcat({''},datestr(dates));

data=[];
for i = 1:length(series)
    a = fetch(link,series{i},startdate,enddate).Data(:,2);
    data = [data, a(:)];
    clear a
end

dataset=mat2dataset(data,'VarNames',{'tres3mo','tres6mo','tres1y','tres2y','tres3y','tres5y',...
    'tres7y','tres10y','tres20y','tres30y'},'ObsNames',NameObs);
clear startdate enddate url series link

maturities = [3, 6, 12, 24, 36, 60, 84, 120, 240, 360]';  % Cast to a column vector
tt = rmmissing(timetable(dates, data));
Yields = tt.data;  % In-sample yield series for estimation
dates = tt.dates;      % Date stamps

%% Two-Step Estimation of the Diebold-Li Model
lambda0 = 0.0609;
X = [ones(size(maturities)) (1-exp(-lambda0*maturities))./(lambda0*maturities) ...
    ((1-exp(-lambda0*maturities))./(lambda0*maturities)-exp(-lambda0*maturities))];

Beta = zeros(size(Yields,1),3);
Residuals = zeros(size(Yields,1),numel(maturities));

for i = 1:size(Yields,1)
    EstMdlOLS = fitlm(X,Yields(i,:)','Intercept',false);
    Beta(i,:) = EstMdlOLS.Coefficients.Estimate';
    Residuals(i,:) = EstMdlOLS.Residuals.Raw';
end

MdlVAR = varm(3,1);
EstMdlVAR = estimate(MdlVAR,Beta);

%% SSM Estimation of the Diebold Li model
Mdl = ssm(@(params)Example_DieboldLi(params,Yields,maturities));

A0 = EstMdlVAR.AR{1};      
A0 = A0(:);                
Q0 = EstMdlVAR.Covariance; 
B0 = [sqrt(Q0(1,1)); 0; 0; sqrt(Q0(2,2)); 0; sqrt(Q0(3,3))];
H0 = cov(Residuals);       
D0 = sqrt(diag(H0));       
mu0 = mean(Beta)';
param0 = [A0; B0; D0; mu0; lambda0];

options = optimoptions('fminunc','MaxFunEvals',25000,'algorithm','quasi-newton', ...
    'TolFun',1e-8,'TolX',1e-8,'MaxIter',1000,'Display','off');

[EstMdlSSM,params] = estimate(Mdl,Yields,param0,'Display','off', ...
    'options',options,'Univariate',true);

lambda = params(end);        % Estimated decay rate    
mu = params(end-3:end-1)';   % Estimated factor means

%% Forecast Estimated SSM
% Inferred Factors
intercept = EstMdlSSM.C*mu';
DeflatedYields = Yields - intercept';
DeflatedStates = smooth(EstMdlSSM,DeflatedYields);
EstimatedStates = DeflatedStates + mu;

% Deterministic MMSE (Minimum Mean Square Error) Forecasts
fh = 12;   % Forecast horizon (months)

[ForecastedDeflatedYields,FMSE] = forecast(EstMdlSSM,fh,DeflatedYields);
MMSEForecasts = ForecastedDeflatedYields + intercept';
% The forecasted yield curves MMSEForecasts is a 12-by-17 matrix; each row corresponds to a period 
% in the forecast horizon and each column corresponds to a maturity.

% Plot of the 1-year ahead forecast
figure
plot(maturities,[MMSEForecasts(fh,:)'])
title('12-Month-Ahead MMSE Forecast')
xlabel('Maturity (Months)')
ylabel('Percent')
legend({'MMSE'},'Location','best')
