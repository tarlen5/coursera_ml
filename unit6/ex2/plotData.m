function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

pos_index = find(y==1);
neg_index = find(y==0);

% Quick Diagnostic check:
%printf("X: ");
%X
%printf("positive y: ");
%X(pos_index,1)
%printf("negative y: ");
%X(neg_index,1)

plot(X(pos_index,1), X(pos_index,2), 'k+', 'LineWidth', 2, 'MarkerSize', 6);
plot(X(neg_index,1), X(neg_index,2), 'ko', 'MarkerFaceColor', 'y',
     'LineWidth', 2, 'MarkerSize',6);


hold off;

end
