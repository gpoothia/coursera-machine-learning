function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


sigmoid_input = X*theta;
h = sigmoid(sigmoid_input);
first_term = -y.*log(h);
one_minus_y = arrayfun(@(x) 1-x, y); 
log_one_minus_h = arrayfun(@(x) log(1 - x), h); 
second_term = one_minus_y .* log_one_minus_h;
error = first_term - second_term;
J_first_term = 1/m*sum(error);
J_second_term = lambda/(2*m) * (sum(theta.*theta) - theta(1)*theta(1));
J = J_first_term + J_second_term;

for j = 1:size(theta)
  j_feature = X(:,j:j);
  term_to_sum = (h - y).*j_feature;
  grad(j) = 1/m*sum(term_to_sum);
  if (j >= 2)
    grad(j) = grad(j) + (lambda/m)*theta(j);
  endif
endfor





% =============================================================

grad = grad(:);

end
