function dataset = load_glc_real_curve_data(inputDir)
%LOAD_GLC_REAL_CURVE_DATA Load and clean UK real forward curve data.
%   DATASET = LOAD_GLC_REAL_CURVE_DATA(INPUTDIR) reads the Bank of England
%   GLC real daily forward curve spreadsheets stored in INPUTDIR, aligns
%   the maturities across vintages, removes all-missing rows, and returns a
%   struct containing the cleaned timetable and maturity grids.

arguments
    inputDir (1, 1) string
end

requiredFiles = [
    "GLC Real daily data_1985 to 1989.xlsx"
    "GLC Real daily data_1990 to 1994.xlsx"
    "GLC Real daily data_1995 to 1999.xlsx"
    "GLC Real daily data_2000 to 2004.xlsx"
    "GLC Real daily data_2005 to 2015.xlsx"
    "GLC Real daily data_2016 to present.xlsx"
];

for fileIndex = 1:numel(requiredFiles)
    filePath = fullfile(inputDir, requiredFiles(fileIndex));
    if ~isfile(filePath)
        error("load_glc_real_curve_data:MissingFile", ...
            "Required input file not found: %s", filePath);
    end
end

variableNamesShort = {'months','30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300'};
variableNamesLong = {'months','30','36','42','48','54','60','66','72','78','84','90','96','102','108','114','120','126','132','138','144','150','156','162','168','174','180','186','192','198','204','210','216','222','228','234','240','246','252','258','264','270','276','282','288','294','300','306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456','462','468','474','480'};

datasets = cell(1, 6);

datasets{1} = readtable(fullfile(inputDir, requiredFiles(1)), ...
    'Sheet', '2.  real fwd curve', 'Range', 'A6:AU1309');
datasets{1}.Properties.VariableNames = variableNamesShort;

datasets{2} = readtable(fullfile(inputDir, requiredFiles(2)), ...
    'Sheet', '2.  real fwd curve', 'Range', 'A6:AU1310');
datasets{2}.Properties.VariableNames = variableNamesShort;

datasets{3} = readtable(fullfile(inputDir, requiredFiles(3)), ...
    'Sheet', '2.  real fwd curve', 'Range', 'A6:AU1310');
datasets{3}.Properties.VariableNames = variableNamesShort;

datasets{4} = readtable(fullfile(inputDir, requiredFiles(4)), ...
    'Sheet', '2.  real fwd curve', 'Range', 'A6:AU1310');
datasets{4}.Properties.VariableNames = variableNamesShort;

datasets{5} = readtable(fullfile(inputDir, requiredFiles(5)), ...
    'Sheet', '2. fwd curve', 'Range', 'A6:AU2874');
datasets{5}.Properties.VariableNames = variableNamesShort;

datasets{6} = readtable(fullfile(inputDir, requiredFiles(6)), ...
    'Sheet', '2. fwd curve', 'Range', 'A6:BY1809');
datasets{6}.Properties.VariableNames = variableNamesLong;
datasets{6} = removevars(datasets{6}, ...
    {'306','312','318','324','330','336','342','348','354','360','366','372','378','384','390','396','402','408','414','420','426','432','438','444','450','456','462','468','474','480'});

combinedData = vertcat(datasets{:});
combinedData = combinedData(~all(ismissing(combinedData(:, 2:end)), 2), :);

dates = datetime(table2array(combinedData(:, 1)), 'InputFormat', 'dd-MM-yyyy');
yieldMatrix = table2array(combinedData(:, 5:end));

maturitiesMonths = [48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264,270,276,282,288,294,300]';
maturitiesYears = maturitiesMonths / 12;

yieldCurve = timetable(dates, yieldMatrix, 'VariableNames', {'yields'});
yieldCurve(yieldCurve.dates < datetime(2000, 5, 24), :) = [];

dataset = struct( ...
    'yield_curve', yieldCurve, ...
    'maturities_months', maturitiesMonths, ...
    'maturities_years', maturitiesYears);

end
