function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

  % Useful variables
  [m, n] = size(X);  % 307 x 2

  % You should return these values correctly
  mu = zeros(n, 1);
  sigma2 = zeros(n, 1);

  % totally vectorized mean:
  mu = 1/m*sum(X,axis=1)';

  
  %---for loop based sigma2---
  %for j=1:m
  %  temp = (X(j,:) - mu').^2;
  %  sigma2 += temp';
  %end
  %sigma2 = 1/m*sigma2

  %---vectorized sigma2---
  sigma2 = 1/m*sum((X - mu').^2,axis=1)';
  
end
