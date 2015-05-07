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

xTheta = X*theta;
J = 1.0/m * (-y'*log(sigmoid(xTheta)) - (1.0 - y)'*log(1.0 - sigmoid(xTheta)));
grad = 1.0/m * ((sigmoid(xTheta) - y)'*X);

% Note: the following three lines are a more verbose way of expressing `grad`
%grad(1) = 1.0/m * ((sigmoid(xTheta) - y)'*X(:,1))
%grad(2) = 1.0/m * ((sigmoid(xTheta) - y)'*X(:,2))
%grad(3) = 1.0/m * ((sigmoid(xTheta) - y)'*X(:,3))

end
