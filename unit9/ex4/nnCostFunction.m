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

% Reshape nn_params back into the parameters Theta1 and Theta2, the
% weight matrices for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size *
				   (input_layer_size + 1))):end),
                   num_labels, (hidden_layer_size + 1));

  % Number of training examples:
  m = size(X, 1);

  % You need to return the following variables correctly
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  debug = 0;
  % Loop for backpropagation algorithm:
  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));
  for i = 1:m
    A1 = X(i,:);
    A1 = [1 A1];
    y_i = y(i,:);
    if debug
      size(A1)     % 1 x 401
      size(y_i)     % 1 x 1
      disp(y_i)     % = 10 (concretely represented as zero).
      size(Theta1)  % 25 x 401
      size(Theta2)  % 10 x 26
      pause;
    end

    z2 = Theta1*A1';
    A2 = sigmoid(z2); % 25x1 (col vector)
    A2 = [1; A2];

    z3 = Theta2*A2;
    A3 = sigmoid(z3);

    y_vec = zeros(num_labels,1);
    y_vec(y_i) = 1;
    cost_i = -y_vec'*log(A3) - (1 - y_vec')*log(1 - A3);

    J += cost_i;

    % Now for the backpropagation specific part:
    delta3 = A3 - y_vec;       % 10 x 1

    z2 = [1; z2];  % This needed??
    delta2_ = Theta2'* delta3 .* sigmoidGradient(z2);
    % Must skip bias layer:
    delta2 = delta2_(2:end);   % 25 x 1

    Delta1 += delta2*A1;
    Delta2 += delta3*A2';

  endfor

  J /= m;

  % Add regularization:
  th1 = Theta1(:,2:end);    % 25 x 400
  th2 = Theta2(:,2:end);    % 10 x 25
  J += lambda/(2*m) * (sum(sumsq(th1,2),1) + sum(sumsq(th2,2),1));

  Theta1_grad = 1/m*Delta1;
  Theta2_grad = 1/m*Delta2;

  Theta1_grad(:,2:end) += lambda/m*Theta1(:,2:end);
  Theta2_grad(:,2:end) += lambda/m*Theta2(:,2:end);

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
