function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

A2 = sigmoid(X * Theta1');
A2 = [ones(m, 1) A2];

A3 = sigmoid(A2 * Theta2');
y_pred = A3;

y_onehot = [zeros(m, num_labels)];
y_onehot(sub2ind(size(y_onehot), 1:m, y')) = 1;

J = (1 / m) * sum(   sum(-y_onehot.*log(y_pred), 2) - sum((1-y_onehot).*log(1-y_pred), 2)  , 1);

% -------------------------------------------------------------
theta_temp1 = Theta1(:,2:size(Theta1,2));
theta_temp2 = Theta2(:,2:size(Theta2,2));

reg = (lambda / (2*m)) * (sum(sum(theta_temp1.^2, 2), 1) + sum(sum(theta_temp2.^2, 2), 1));

J = J + reg;

% Back propagation
for t=1:m
    sample_A1 = X(t, :); # 1x401

    sample_Z2 = sample_A1 * Theta1'; # 1x401 * (25x401)' = 1x25
    sample_A2 = sigmoid(sample_Z2);  # 1x25
 
    sample_A2 = [1 sample_A2]; # 1x26

    sample_Z3 = sample_A2 * Theta2'; # 1x26 * (10x26)' = 1x10
    smaple_A3 = sigmoid(sample_Z3); # 1x10


    error_l3 = smaple_A3 - y_onehot(t, :); # 1x10 - 1x10
    sample_Z2 = [1 sample_Z2]; # 1x26
    
    error_l2 = (error_l3 * Theta2) .* sigmoidGradient(sample_Z2); # (1x10 * 10x26) .* 1x26
    error_l2 = error_l2(2:end); # 1x25

    % fprintf('%dx%d\n', size(Theta1_grad));

    Theta2_grad = Theta2_grad + error_l3' * sample_A2; # 10x26 + (1x10)' * 1x26 = 10x26 + 10x26
	Theta1_grad = Theta1_grad + error_l2' * sample_A1; # 25x401 + (1x25)' * 1x401 = 25x401 + 25x401
    
end;

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
