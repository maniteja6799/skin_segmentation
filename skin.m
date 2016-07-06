%this is the main file , this needs plotData.m, costFunctionReg.m, 
%
%
%
%% Initialization
clear ; close all; clc

%% Load Data
%  The first three columns contains the X values and the third column
%  contains the label (y).

data = load('Skin_NonSkin.txt');
data = data(randperm(size(data,1)),:);
X = data(1:180000, [1,2,3]); y = double((data(1:180000, 4) == 1));
plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('R')
ylabel('G')
zlabel('B')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% feature mapping and Cost function

%mapping attributes to higher dimensions
X = mapFeature(X(:,1), X(:,2), X(:,3));

%X = normalise(X);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 10;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% training

initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 10;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 200);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options)

