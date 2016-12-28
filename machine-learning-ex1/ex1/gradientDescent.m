function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%X dim are m + 1 x 2
%pureX dim are  m x 1
pureX = X(:,2:2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %delta = zeros(2,1);
    % h dim are m x 1
    %h = X*theta; 
    %h_minus_y = h - y;
    %delta_vec = 1/m*(h_minus_y' * pureX);
    %theta = theta - alpha*delta; 

    temp1 = (X*theta - y);
    newtheta1 = theta(1) - alpha*(1/m*sum(temp1));

    temp2 = (X*theta - y);
    newtheta2 = theta(2) - alpha*(1/m*sum(temp2' * pureX));

    theta(1) = newtheta1;
    theta(2) = newtheta2;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
