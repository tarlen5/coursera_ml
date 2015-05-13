function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

  % Number of training examples
  m = size(X, 1);
  m_cv = size(Xval,1);

  % You need to return these values correctly
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);

  lambda_train = 0
  theta = [1 ; 1];
  for i=1:m
    X_i = X(1:i,:);
    y_i = y(1:i);
    
    % 1) Train linear regression on just i samples of training set
    [theta] = trainLinearReg([ones(i, 1) X_i], y_i, lambda);
    
    [J_train, grad] = linearRegCostFunction([ones(i, 1) X_i], y_i, theta, 
					    lambda_train);
    % 2) Compute the error on this training set
    error_train(i) = J_train;
    
    % 3) Compute the error on the cross-validation set
    [J_val, grad] = linearRegCostFunction([ones(m_cv, 1) Xval], yval, theta, 
					  lambda_train);
    error_val(i) = J_val;
    
    
  endfor


end
