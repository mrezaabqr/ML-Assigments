function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% fprintf('%dx%d\n', size(X));
% fprintf('%dx%d\n', size(theta));

y_pred =  X * theta;

% fprintf('%dx%d\n', size(y_pred));
% fprintf('%dx%d\n', size(y));

J = (1/m) * sum((y - y_pred).^2);

% =========================================================================

end
