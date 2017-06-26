function [ theta,J_history ] = gradientDescentMulti( X, y, theta, alpha, num_iters )
%GRADIENTDESCENTMULTI 此处显示有关此函数的摘要
%   此处显示详细说明
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
theta = theta - (alpha / m) * (X' * ((X * theta) - y));
J_history(iter) = costfunction(X, y, theta);

end

end

