clear; clc; close all; format compact;
set(0,'defaultTextInterpreter','latex'); %trying to set the default
set(groot, 'defaultAxesTickLabelInterpreter','latex');% set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultFigureUnits', 'pixels', 'defaultFigurePosition', [440   500   670   225]);
set(0,'defaultAxesFontSize',24);
set(0, 'DefaultLineLineWidth', 2);

run_time = [1.676,           2.350,          3.174,   1.019, 1.000, 2.278, 12.029, 1.128,1.093,1.056 ];
method_ = ["SplitConformal", "NExConformal", "FACI", "SF-OGD", "SimpleOGD", "FACI-S",  "SAOCP", "Mag. Learner", "Undiscount. Mag. Learner", "Mag. Learner ($h_t = 0$)"];
[run_time, idx] = sort(run_time);
run_time = run_time/run_time(1);
method_ = method_(idx);

figure
bar(method_,run_time)
Y = run_time;
text(1:length(Y),Y,num2str(round(Y',2)),'vert','bottom','horiz','center', 'FontSize',24); 
ylabel("Runtime (s)")