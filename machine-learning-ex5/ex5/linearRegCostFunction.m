function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
sqrErr = (h - y).^2;
first_term = 1/(2*m)*sum(sqrErr);
second_term = lambda/(2*m) * (sum(theta.*theta) - theta(1)*theta(1));
J = first_term + second_term;

for j = 1:size(theta)
  j_feature = X(:,j:j);
  term_to_sum = (h - y).*j_feature;
  grad(j) = 1/m*sum(term_to_sum);
  if (j >= 2)
    grad(j) = grad(j) + (lambda/m)*theta(j);
  endif
endfor








% =========================================================================

grad = grad(:);

end
