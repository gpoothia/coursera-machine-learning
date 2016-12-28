
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

%% For all inputs find network predictions h(theta)    
[a1, z2, a2, z3, a3] = mypredict(Theta1, Theta2, X);

%for i = 1:1
for i = 1:m
  %for k = 1:1
  for k = 1:num_labels
    if (y(i) == k)
      yki = 1; 
    else
      yki = 0;

    endif 
    first_term = -yki*log(a3(k,i));
    one_minus_y = arrayfun(@(x) 1-x, yki); 
    log_one_minus_h = arrayfun(@(x) log(1 - x), a3(k,i)); 
    second_term = one_minus_y .* log_one_minus_h;
    error = first_term - second_term;
    J = J + error;
  endfor
endfor

J = 1/m*J;

reg_term1 =0;
reg_term2 = 0;
for j =1:rows(Theta1)
  for k = 2:columns(Theta1)
    reg_term1 = reg_term1 + Theta1(j,k)**2;
  endfor
endfor
for j =1:rows(Theta2)
  for k = 2:columns(Theta2)
    reg_term2 = reg_term2 + Theta2(j,k)**2;
  endfor
endfor

J = J + (reg_term1 + reg_term2)*lambda/(2*m);

% Backpropogation start

%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
bin_y = zeros(num_labels, m); 
for i = 1:m
  bin_y(y(i), i) = 1;
endfor

big_delta1 = zeros(rows(Theta1),columns(Theta1)); 
big_delta2 = zeros(rows(Theta2),columns(Theta2)); 
for i = 1:m
  % we have already found a1, a2, a3 for all inputs 
  yi = bin_y(:,i);
  a3i = a3(:,i);
  small_delta3 = a3i .- yi; 
  z2i = z2(:,i);
  %size(z2i)
  term1 = Theta2'*small_delta3;
  % remove top row to reduce from 26 x1 to 25 x1
  term1 = term1(2:end, :);
  term2 = sigmoidGradient(z2i);

  small_delta2 = term1 .* term2;
  %size(small_delta2)
  
  a1i = a1(i,:);
  big_delta1 = big_delta1 + small_delta2*a1i;
  %temp = small_delta2*a1i';
  %if (m == 1)
  %  big_delta1 = temp;
  %else
  %  big_delta1 = big_delta1 + temp;
  %endif
  a2i = a2(:,i);
  big_delta2 = big_delta2 + small_delta3*a2i';
endfor
Theta1_grad = (1/m).*big_delta1;
Theta2_grad = (1/m).*big_delta2;

for i = 1:rows(Theta1)
  for j = 1:columns(Theta1)
  if j != 1
    Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
  endif
  endfor
endfor
for i = 1:rows(Theta2)
  for j = 1:columns(Theta2)
  if j != 1
    Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i,j);
  endif
  endfor
endfor

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [a1, z2, a2, z3, a3]  = mypredict(Theta1, Theta2, X)
  %Add a first column of all 1s
  a1=[ones(size(X,1),1) X]; 
  z2 = Theta1*a1';
    %Add row of all 1s
    %ones_row = ones(1,columns(z2));
    %z2 = [ones_row; z2];
  a2 = sigmoid(z2);
  %Add row of all 1s
  ones_row = ones(1,columns(a2));
  a2 = [ones_row; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3); 
end


