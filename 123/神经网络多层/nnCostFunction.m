function [J grad] = nCostFunction(nn, in, h1, ...
                   h2,ou, X, y, lambda);
Theta1 =reshape(nn(1:(in+1)*h1),h1,(in+1));
Theta2 =reshape(nn(((in+1)*h1+1):(((in+1)*h1)+h1*(h2+1))),h2,(h1+1));
Theta3 =reshape(nn((1+(((in+1)*h1)+h1*(h2+1))):end),ou,(h2+1));
%这里的修改一定要注意具体的要不要+1,111到500.而不是501
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta3_grad = zeros(size(Theta3));

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
n = size(a2,1);
a2 = [ones(n,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
p = size(a3,1);
a3 = [ones(p,1) a3];
z4 = a3 * Theta3';
a4 = sigmoid(z4);

for k = 1:nn
	y_k = (y == k);
	a4_k = a4(:,k);
	J_K =1/m * sum(-y_k .* log(a4(:,k)) - (1 - y_k) .* log(1 - a4(:,k)));
	J = J + J_K;
end
%part2

J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

%______________________________________________________________________
for t = 1:m
	for k = 1:ou
		y_k = (y(t) == k);
		delta4(k,1) = a4(t,k) - y_k;
    end
    delta3 = Theta3(:,2:end)' * delta4 .* sigmoidGradient(z3(t,:)');
    delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2(t,:)');
	Theta1_grad = Theta1_grad + delta2 * a1(t,:);
	Theta2_grad = Theta2_grad + delta3 * a2(t,:);
    Theta3_grad = Theta3_grad + delta4 * a3(t,:);
end


Theta1_grad(:,1) = 1/m .* Theta1_grad(:,1);
Theta1_grad(:,2:end) = 1/m .* Theta1_grad(:,2:end) + lambda/m .* Theta1(:,2:end);
Theta2_grad(:,1) = 1/m .* Theta2_grad(:,1);
Theta2_grad(:,2:end) = 1/m .* Theta2_grad(:,2:end) +lambda/m .* Theta2(:,2:end);
Theta3_grad(:,1)=1/m .* Theta3_grad(:,1);
Theta3_grad(:,2:end) = 1/m .* Theta3_grad(:,2:end)+lambda/m .* Theta3(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:);Theta3_grad(:)];



end


