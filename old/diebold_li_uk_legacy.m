%% Apply State-Space Methodology to Analyze Diebold-Li Yield Curve Model 
% SMM943 Financial Econometrics - Coursework      
% Author: Emilio Luis Sáenz Guillén, PhD Student in Actuarial Science  
% Professor: Giovanni Urga                                         
% Bayes Business School; December 2022 

clc
clear
cd 'C:\Users\addj700\Dropbox\Bayes Business School\PhD\Modules\1st Term\Financial Econometrics (SMM943)\Presentation - ML\Empirical application\input\glcrealddata';

VariableNames_1 = {'months','30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300'};
VariableNames_2 = {'months','30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300','306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456','462','468','474','480'};


data_1 = readtable('GLC Real daily data_1985 to 1989.xlsx','Sheet','2.  real fwd curve','Range','A6:AU1309');
data_1.Properties.VariableNames = VariableNames_1;
data_2 = readtable('GLC Real daily data_1990 to 1994.xlsx','Sheet','2.  real fwd curve','Range','A6:AU1310');
data_2.Properties.VariableNames = VariableNames_1;
data_3 = readtable('GLC Real daily data_1995 to 1999.xlsx','Sheet','2.  real fwd curve','Range','A6:AU1310');
data_3.Properties.VariableNames = VariableNames_1;
data_4 = readtable('GLC Real daily data_2000 to 2004.xlsx','Sheet','2.  real fwd curve','Range','A6:AU1310');
data_4.Properties.VariableNames = VariableNames_1;
data_5 = readtable('GLC Real daily data_2005 to 2015.xlsx','Sheet','2. fwd curve','Range','A6:AU2874');
data_5.Properties.VariableNames = VariableNames_1;
data_6 = readtable('GLC Real daily data_2016 to present.xlsx','Sheet','2. fwd curve','Range','A6:BY1809');
data_6.Properties.VariableNames = VariableNames_2;
% remove from data_6 the maturities that are not present in the other tables:
data_6= removevars(data_6,{'306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456',	'462','468','474','480'});

data = [data_1; data_2; data_3; data_4; data_5; data_6];
data = data(~all(ismissing(data(:,2:end)),2),:);

clear VariableNames_1 VariableNames_2

dates = datetime(table2array(data(:,1)),'InputFormat','dd-MM-yyyy');
data = table2array(data(:,5:end)); % not include maturity = 30, 36, 42 months as they have too many NaNs

maturities =[48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264,270,276,282,288,294,300]';
maturities_in_years = [4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25]';

tt = timetable(dates, data); 
tt(tt.dates < "24-May-2000", :) = [];


%% Surf plot
figure
    surf(maturities_in_years,[tt.dates], [tt.data]),
    datetick('y','yyyy'), 
    ylabel('Date'),
    xlabel('Maturities (in years)'),
    colormap winter
%% Diebold-Li
MMSEForecasts=[];
for i= 0:254
    data = [data_1; data_2; data_3; data_4; data_5; data_6];
    data = data(~all(ismissing(data(:,2:end)),2),:);
    dates = datetime(table2array(data(:,1)),'InputFormat','dd-MM-yyyy');
    data = table2array(data(:,5:end)); % not include maturity = 30, 36, 42 months as they have too many NaNs
    tt = timetable(dates, data); 
    tt(tt.dates < "24-May-2000", :) = [];
    tt(5439+i:end,:) = []; % Select training sample

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
    fh = 1;   % Forecast horizon (days)

    [ForecastedDeflatedYields,FMSE] = forecast(EstMdlSSM,fh,DeflatedYields);
    MMSEForecasts = [MMSEForecasts; ForecastedDeflatedYields + intercept'];
end

% The forecasted yield curves MMSEForecasts is a matrix; each row corresponds to a period 
% in the forecast horizon and each column corresponds to a maturity.

% Plot of the forecast
fh= 12; % periods ahead
figure
plot(maturities_in_years,MMSEForecasts(fh,:)')
title('255 days-ahead MMSE forecast')
xlabel('Maturity (Years)')
ylabel('Percent')
legend({'MMSE'},'Location','best');

cd 'C:\Users\addj700\Dropbox\Bayes Business School\PhD\Modules\1st Term\Financial Econometrics (SMM943)\Presentation - ML\Empirical application\output';
writematrix(MMSEForecasts,'MMSEForecasts.csv') 

