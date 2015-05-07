function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
%   NOTE: z can be a matrix, vector or scalar

% You need to return the following variables correctly
g = zeros(size(z));

g = 1./(1.0 + exp(-z));

end
