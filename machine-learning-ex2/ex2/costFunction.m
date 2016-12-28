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


sigmoid_input = X*theta;
h = sigmoid(sigmoid_input);
first_term = -y.*log(h);
one_minus_y = arrayfun(@(x) 1-x, y);
log_one_minus_h = arrayfun(@(x) log(1 - x), h);
second_term = one_minus_y .* log_one_minus_h;
error = first_term - second_term;
J = 1/m*sum(error);

for j = 1:size(theta)
  j_feature = X(:,j:j);
  term_to_sum = (h - y).*j_feature;
  grad(j) = 1/m*sum(term_to_sum);
endfor



% =============================================================

end
