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

%disp(size(theta)); % 2 x 1
%disp(size(X));     % 12 x 2
%disp(size(y));     % 12 x 1

hTheta = X*theta;
J = 1/(2.0*m) * (hTheta - y)'*(hTheta - y) + lambda/(2.0*m)*(theta(2:end)'*theta(2:end));

grad = 1/m*(hTheta - y)'*X;
grad(2:end) += lambda/m*theta(2:end)';


grad = grad(:);

end
