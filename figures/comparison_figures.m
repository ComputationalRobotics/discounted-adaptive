%% Set the default font to Computer Modern LaTeX
clear; clc; close all;
set(groot, 'DefaultLegendInterpreter', 'latex');
set(0,'defaultTextInterpreter','latex'); %trying to set the default
set(groot, 'defaultAxesTickLabelInterpreter','latex');% set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultFigureUnits', 'pixels', 'defaultFigurePosition', [440   500   670   400]);
set(0,'defaultAxesFontSize',24);
set(0, 'DefaultLineLineWidth', 3);

%% Read the CSV file and organize data
df = readtable('/Users/davidbombara/Documents/ComputationalRoboticsCode/discounted-adaptive/figures/grouped_data_sudden_method.csv');
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


%% Plot Coverage

plot_data(cov, [0.89, 0.91], "Average Coverage");


plot_data(avg_width, [120, 150], "Average Width");

plot_data(avg_miscov, [0.03, 0.05], "Average Miscoverage");

%title(han,'yourTitle');

plot_data(avg_regret, [0.0, 0.021], "Average Regret");

function plot_data(y_data, y_limit, y_label_str)
    D = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3];

    % Show results for:
    % - Scale-Free OGD (4)
    % - Simple OGD (5) 
    % - SACOP (7)
    % - Magnitude Learner (8)
    % - Undiscounted Magnitude Learner (9)
    idx_toplot = [4,5,7,8,9];
    learning_methods = ["Split Conformal", "NExConformal", "FACI", "Scale-Free OGD", "Simple OGD", "FACI-S", "SAOCP", "Magnitude Learner", "Undiscounted Magnitude Learner"];
    fig = figure;
    subplot(2,1,1);
    for i = idx_toplot
        semilogx(D, y_data{i},'o-','MarkerSize',10);
        hold on
        xlim([1e-3,1e3])
    end
    if strcmpi(y_label_str,"Average Coverage")
        legend(learning_methods(idx_toplot))
    end
    %title("Average Regret")
    title("Radius Prediction for TinyImageNet")
    subplot(2,1,2)
    for i = idx_toplot
        semilogx(D,y_data{i},'o-','MarkerSize',10);
        hold on
        %ylim([0.0,0.021])
        ylim(y_limit)
        xlim([1e-3,1e3])
    end
    xlabel('$D_{est} / D_{actual}$');
    
    han=axes(fig,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    %ylabel(han,'Average Regret');
    ylabel(han, y_label_str);
end
