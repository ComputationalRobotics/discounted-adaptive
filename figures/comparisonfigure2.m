%% Set the default font to Computer Modern LaTeX
clear; clc; close all;
set(groot, 'DefaultLegendInterpreter', 'latex');
set(0,'defaultTextInterpreter','latex'); %trying to set the default
set(groot, 'defaultAxesTickLabelInterpreter','latex');% set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultFigureUnits', 'pixels', 'defaultFigurePosition', [440   500   670   400]);
set(0,'defaultAxesFontSize',16);
set(0, 'DefaultLineLineWidth', 2);

%% Read the CSV file and organize data
df = readtable('/figures/grouped_data_sudden_method.csv');
metrics = ["Coverage", "Avg. Width", "Avg. Miscoverage", "Avg. Regret"];
methods_abbr = ["SplitConformal","NExConformal","FACI", "ScaleFreeOGD", "SimpleOGD", "FACI_S", "SAOCP", "MagnitudeLearner","MagLearnUndiscounted"];
D = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3];

idx = cell(length(methods_abbr),1);
cov = cell(length(methods_abbr),1);
avg_width = cell(length(methods_abbr),1);
avg_regret = cell(length(methods_abbr),1);
avg_miscov = cell(length(methods_abbr),1);
for i = 1:length(methods_abbr)
    idx{i} = strcmp(df.Method, methods_abbr(i));
    cov{i} = df.Cov(idx{i});
    avg_width{i} = df.AvgWidth(idx{i});
    avg_regret{i} = df.AvgRegret(idx{i});
    avg_miscov{i} = df.AvgMiscov(idx{i});
end
idx = strcmpi(df.Method,'SplitConformal');
idx_toplot = [4,5,7,8,9];

%% Plotting
% Style: 3 rows, 2 columns of subplots
% legend outside of figures, below or above
% Column 1: y-axis shows entire data
% Column 2: y-axis is zoomed in
% Row 1: Average coverage
% Row 2: Average Width
% Row 3: Average regret
learning_methods = ["Split Conformal", "NExConformal", "FACI", "Scale-Free OGD", "Simple OGD", "FACI-S", "SAOCP", "Magnitude Learner", "Undiscounted Magnitude Learner"];
    
fig = figure;

subplot(3,2,1); % Row 1, Column 1: Average coverage, full data
for i = idx_toplot
    semilogx(D, cov{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3,1e3])
end
ylabel({"Average";"Coverage"})

subplot(3,2,3); % Row 2, Column 1: Average width, full data
for i = idx_toplot
    semilogx(D, avg_width{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3,1e3])
end
ylabel({"Average"; "Width"})
subplot(3,2,5); % Row 3, Column 1: Average regret, full data
for i = idx_toplot
    semilogx(D, avg_regret{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3,1e3])
end

xlabel('$D_{est} / D_{actual}$')
ylabel({"Average"; "Regret"})
%% Zoomed y-axis
%
subplot(3,2,2); % Row 1, Column 2: Average coverage, zoomed y axis
for i = idx_toplot
    semilogx(D, cov{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3,1e3])
    ylim([0.89, 0.91])
end
%
subplot(3,2,4); % Row 2, Column 2: Average width, zoomed y axis
for i = idx_toplot
    semilogx(D, avg_width{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3,1e3])
    ylim([120, 150])
end
%
subplot(3,2,6); % Row 3, Column 2: Average regret, full data
for i = idx_toplot
    semilogx(D, avg_regret{i},'o-','MarkerSize',5);
    hold on
    xlim([1e-3-1e-4,1e3+1e4])
    ylim([0.0, 0.021])
end
xlabel('$D_{est} / D_{actual}$')
legend(learning_methods(idx_toplot), 'NumColumns',5,'Location','southoutside')

han=axes(fig,'visible','off'); 
han.Title.Visible='on';
title(han,"Radius Prediction for TinyImageNet")
% han.XLabel.Visible='on';
% han.YLabel.Visible='on';
%ylabel(han,'Average Regret');
%ylabel(han, y_label_str);