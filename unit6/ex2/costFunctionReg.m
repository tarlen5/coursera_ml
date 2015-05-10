function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

xTheta = X*theta;
thetaSq = theta(2:size(theta))'*theta(2:size(theta));

J = 1.0/m * (-y'*log(sigmoid(xTheta)) - (1.0 - y)'*log(1.0 - sigmoid(xTheta))) + lambda/(2.0*m)*thetaSq;

n = length(grad);
grad(1) = 1.0/m * ((sigmoid(xTheta) - y)'*X(:,1));
grad(2:n) = ( 1.0/m * ((sigmoid(xTheta) - y)'*X(:,2:n)) +
                         lambda/m*theta(2:n)');


end
