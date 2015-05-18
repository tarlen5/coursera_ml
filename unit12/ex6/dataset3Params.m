function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================

CRange = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigmaRange = CRange;

error_min = 1.0;

for i=1:length(CRange)
  Ci = CRange(i)
  for j=1:length(sigmaRange)
    sigmaj = sigmaRange(j);
    model = svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigmaj));
    predictions = svmPredict(model, Xval);
    error_cur = mean(double(predictions ~= yval));

    if error_cur < error_min
      error_min = error_cur;
      C = Ci;
      sigma = sigmaj;
    endif

  endfor
endfor


end
