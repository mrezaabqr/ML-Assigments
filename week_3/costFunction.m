function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% fprintf('%dx%d\n', size(theta));
% fprintf('%dx%d\n', size(X));
% fprintf('%dx%d\n', size(y));

% sigmoid( 100x2 * 2x1)
y_pred = sigmoid(X * theta);

% % 100x1 * log(100x1) - (1 - 100x1) * (1 - 100x1)
J = (1 / m) * sum( -y .* log(y_pred) - (1 - y) .* log(1 - y_pred));

% % 100x2 * (100x1)
grad = (1 / m) * X' * (y_pred - y);
% grad = (1 / m) * X' * (  (-y ./ y_pred) - ((1 - y) ./ (1 - y_pred)) );

% =============================================================

end
