function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = arrayfun(@myfun, z);



% =============================================================

end

function [ out ] = myfun(num)
  %out = num + 5;
  out = 1/(1 + exp(-num));
end
