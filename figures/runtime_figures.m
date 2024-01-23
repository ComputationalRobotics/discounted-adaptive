clear; clc; close all; format compact;
set(0,'defaultTextInterpreter','latex'); %trying to set the default
set(groot, 'defaultAxesTickLabelInterpreter','latex');% set(groot, 'defaultLegendInterpreter','latex');
%set(groot, 'defaultFigureUnits', 'pixels', 'defaultFigurePosition', [440   378   560   700]);
set(groot, 'defaultFigureUnits', 'pixels', 'defaultFigurePosition', [440   500   670   225]);
set(0,'defaultAxesFontSize',24);
set(0, 'DefaultLineLineWidth', 2);

run_time = [2.518867, 3.433757, 4.590309, 1.528370, 1.511978, 3.278640, 17.104501, 1.624969];
method_ = ["SplitConformal", "NExConformal", "FACI", "ScaleFreeOGD", "SimpleOGD", "FACI-S",  "SAOCP", "MagnitudeLearner"];
[run_time, idx] = sort(run_time);
method_ = method_(idx);

figure
bar(method_,run_time)
Y = run_time;
text(1:length(Y),Y,num2str(round(Y',2)),'vert','bottom','horiz','center', 'FontSize',24); 
ylabel("Runtime (s)")

stand_dev  = [
    0.024651;
    0.051077;
    0.048380;
    0.040316;
    0.062259;
    0.055005;
    0.178111;
    0.077406;
    0.059918];

stand_dev = stand_dev(idx);

er = errorbar(method_, run_time, stand_dev, stand_dev);
er.Color = [0,0,0];
er.LineStyle = 'none';
hold off;