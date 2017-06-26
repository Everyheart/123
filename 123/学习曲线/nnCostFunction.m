function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
n = size(a2,1);
a2 = [ones(n,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);


for k = 1:num_labels
	y_k = (y == k);
	a3_k = a3(:,k);
	J_K =1/m * sum(-y_k .* log(a3(:,k)) - (1 - y_k) .* log(1 - a3(:,k)));
	J = J + J_K;
end
%part2

J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


for t = 1:m
	for k = 1:num_labels
		y_k = (y(t) == k);
		delta3(k,1) = a3(t,k) - y_k;
	end
% end
% size(sigmoidGradient(z2))
% size(delta3)
 % size(Theta2)
 % size(delta3)
 % size(a2)

 % pause;
	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2(t,:)');


	% size(delta2)
	% size(a1)
	% size(delta3)
	% size(a2)

	% pause;

	Theta1_grad = Theta1_grad + delta2 * a1(t,:);
	Theta2_grad = Theta2_grad + delta3 * a2(t,:);
end

% size(Theta1_grad)
% size(Theta2_grad)
% pause;

Theta1_grad(:,1) = 1/m .* Theta1_grad(:,1);
Theta1_grad(:,2:end) = 1/m .* Theta1_grad(:,2:end) + lambda/m .* Theta1(:,2:end);
Theta2_grad(:,1) = 1/m .* Theta2_grad(:,1);
Theta2_grad(:,2:end) = 1/m .* Theta2_grad(:,2:end) +lambda/m .* Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
