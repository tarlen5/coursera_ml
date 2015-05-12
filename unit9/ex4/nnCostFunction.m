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
  % Need to add the one to x_i
  for i = 1:m
    x_i = X(i,:);
    x_i = [1 x_i];
    y_i = y(i,:);
    if debug
      size(x_i)     % 1 x 401
      size(y_i)     % 1 x 1
      disp(y_i)     % = 10 (concretely represented as zero).
      size(Theta1)  % 25 x 401
      size(Theta2)  % 10 x 26    
      pause;
    end
    
    A2 = sigmoid(Theta1*x_i'); % 25x1 (col vector)
    A2 = [1; A2];
    
    A3 = sigmoid(Theta2*A2);
  
    y_vec = zeros(num_labels,1);
    y_vec(y_i) = 1;
    cost_i = -y_vec'*log(A3) - (1 - y_vec')*log(1 - A3);
    
    J += cost_i;
    
  endfor
  
  J = J/m;
  
  % Add regularization:
  th1 = Theta1(:,2:end);
  th2 = Theta2(:,2:end);
  %size(th1)  % 25 x 400
  %size(th2)  % 10 x 25

  J += lambda/(2*m) * (sum(sumsq(th1,2),1) + sum(sumsq(th2,2),1));

  
  % Loop for backpropagation algorithm:
  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));
  for t = 1:m
  
    % Forward propagate:
    A1 = X(t,:);
    A1 = [1 A1];  % row vector: 1 x 401
    
    z2 = Theta1*A1';
    A2 = sigmoid(z2);
    A2 = [1; A2];
    
    z3 = Theta2*A2;
    A3 = sigmoid(z3);

    
    y_vec = zeros(num_labels,1);
    y_t = y(t,:);
    y_vec(y_t) = 1;
    delta3 = A3 - y_vec; % 10 x 1
    
    %disp(size(delta3));  % 10 x 1
    %disp(size(A3));      % 10 x 1
    %disp(size(z2));      % 25 x 1
    %disp(size(Theta2));  % 10 x 26
    
    z2 = [1; z2];  % This needed??
    delta2_ = Theta2'* delta3 .* sigmoidGradient(z2);
    % Must skip bias layer:
    delta2 = delta2_(2:end);
    
    %size(delta2)    % 25 x 1
    
    Delta1 += delta2*A1;
    Delta2 += delta3*A2';
  
  endfor
  
  Theta1_grad = 1/m*Delta1;
  Theta2_grad = 1/m*Delta2;
  
  Theta1_grad(:,2:end) += lambda/m*Theta1(:,2:end);
  Theta2_grad(:,2:end) += lambda/m*Theta2(:,2:end);

% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial 
%         derivatives of the cost function with respect to Theta1 and
%         Theta2 in Theta1_grad and Theta2_grad, respectively. After 
%         implementing Part 2, you can check that your implementation is 
%         correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into
%               a binary vector of 1's and 0's to be used with the neural 
%               network cost function.
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




  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
