������
��һ���֣���һ�������򴫲���Ԥ�⣨��֪������
��������������������������������������������������������������������������������
input_layer_size  = 400;  
hidden_layer_size = 25; 
num_labels = 10;                   

load('ex3data1.mat');%�����X��y��ԭ�����������Ѿ����úã���˲���
m = size(X, 1);
sel = randperm(size(X, 1));%����γ�һ�����󣬽��������
sel = sel(1:100);
displayData(X(sel, :));%�����ͼ����һ���ܽ���
load('ex3weights.mat');%���ز���
pred = predict(Theta1, Theta2, X);%Ԥ������������������X��������ǿ������Լ�������ȥ����
mean(double(pred == y)) * 100%��������׼ȷ�ʣ��Ƚ�ÿһ��Ԥ��ĺ�Y,����ֵ
%���ú���------------------------------------------------------------------
function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
alpha1 = [ones(m,1) X];
alpha2 =sigmoid(Theta1*alpha1')';
m = size(alpha2,1);
alpha2 = [ ones(m,1) alpha2];
alpha3 = sigmoid(Theta2*alpha2');%������һ���¸��������Ǹ�����������Ҫ�ҵ����֣������õ�max���Ǹ�û��������ʵ������˵��������������ݣ��ر�ע���йص��Ǹ�����Ĺ���
��������[y,p] = max(alpha3);%�����ҵ��������У�Ȼ��Ԥ��Ľ����������Ҫ�ģ�
p = p';
end
�ڶ����֣�ʹ�÷��򴫲���ѧϰѵ�����������ս���Ԥ��
��һ�����������򴫲����������
����������������������������������������������������������������
load('ex4weights.mat');%�������ֵ����һ�������ϲ���һ���µľ�֤
nn_params = [Theta1(:) ; Theta2(:)];
��1����Ӧ�������õĵ�һ���֣������J��
lambda = 0;%�趨��ʼ��ֵ
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
 ��2����Ӧ�������õĵڶ����֣���������򻯵�J
 lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
 ���������ƫ�����ĺ���2�� g = sigmoidGradient([1 -0.5 0 0.5 1]);Ҳ������һ�����������ƫ����
 ����ʼ������ֵ���ú���3��
 initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
�����ｫ���ٴκϳ�һ������
initial_nn_params = [initial_Theta1(:) ; ini tial_Theta2(:)];
%���ú���1���£�
%��������������������������������������������������������������������������������
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)%��������ô������������ǰֵ
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));%��ԭ���Ĳ����ָ���ԭ���Ĳ�����ʽ����Ϊ����Ӧ�õĲ���
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));%���涼�ǳ�ʼ��ֵ������
%����������������������������������������������������
��1�����򴫲��������cost
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
n = size(a2,1);
a2 = [ones(n,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);%����һ������Ĵ�����ʵ�����������Ҫ��a3
%����һ���ǱȽϸ��ӵ�
for k = 1:num_labels%��1����K����ǩ
�ص�y_k = (y == k);������y�е�ÿһ��λ�õ���K�Ƚϣ����һ������ô����1 ��
������ص��ǣ�y�Ǹ����ģ����ľ���0-10����ô�����Ľ���������ǵõ��˶�Ӧ������������ʾy-kҲ����
��k����ǩ����Ӧ���������������
	a3_k = a3(:,k);%�����ǵõ��˸���K�ľ����ֵ������ֻҪ��K�У��ҵ���Ԥ��Ϊk������
	J_K =1/m * sum(-y_k .* log(a3(:,k)) - (1 - y_k) .* log(1 - a3(:,k)));
    %������ͳ���ʱ��ĵ�K��ǩ�µ�j
	J = J + J_K;%��͵õ��ܵ�j
end
%������������������������������������������������������������
��2�����򻯺���������򻯵�����
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
%������������������������������������������������������������������������������������
��3�������Ӧ��theta1��theta2 ��ƫ����
for t = 1:m%ÿ��������Ҫȥ���������
	for k = 1:num_labels
		y_k = (y(t) == k);%�жϵ�t�����������ĸ���ǩ
		delta3(k,1) = a3(t,k) - y_k;%ֱ�Ӽ�����������Ҹ��Ƶ����������
    end
	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2(t,:)');
    %�������ù�ʽ������ڶ��������е����
    Theta1_grad = Theta1_grad + delta2 * a1(t,:);%������Ƿ��򴫲���ƫ������һ��
	Theta2_grad = Theta2_grad + delta3 * a2(t,:);%
end
% ���������ƫ�����ľ���J=0�벻����0����������빫ʽ����Ҫ��ѧ��⡣
��������޸ĺ�Ĺ�ʽ��������
Theta1_grad(:,1) = 1/m .* Theta1_grad(:,1);
Theta1_grad(:,2:end) = 1/m .* Theta1_grad(:,2:end) + lambda/m .* Theta1(:,2:end);
Theta2_grad(:,1) = 1/m .* Theta2_grad(:,1);
Theta2_grad(:,2:end) = 1/m .* Theta2_grad(:,2:end) +lambda/m .* Theta2(:,2:end);
=========================================
��4��
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
%���ú���2����������������������������������������������������������������������������
function g = sigmoidGradient(z)%�����������ƫ����
g = zeros(size(z));
g = sigmoid(z) .* (1 - sigmoid(z));
end
%���ú���3������������������������������������������������������������������������
function W = randInitializeWeights(L_in, L_out)%����������еĳ�ʼ����������
W = zeros(L_out, 1 + L_in);%���Ի��������ǹ�ʽ
epsilon_init= 0.12;%�������Ŀ���ģ�һ�㲻��̫��
W = rand(L_out,1 + L_in) * 2 * epsilon_init - epsilon_init;%ʵ������γ�һ���󣬺���epsonlon����
end
����������������������������������������������������������������������������������������������������������������������������
�ڶ�������ƫ������ȷ�Եķ����������ֲ�ͬ�ķ���ȥ�󣬹�ʽ�Ͷ���ķ�ʽȥ��ȥ��֤
checkNNGradients;
%���ú���������������������������������������������������������������������������
function checkNNGradients(lambda)
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
%������С��������ȥ�����������Ҳ�����������������ھ����õ�ʱ���ٻ����ο�
%function W = debugInitializeWeights(fan_out, fan_in)
%function numgrad = computeNumericalGradient(J, theta)
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';
nn_params = [Theta1(:) ; Theta2(:)];
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);
[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
end
_____________________________________________________________
�������������޸ĺ�
lambda = 3;
checkNNGradients(lambda);%�ٴμ���������󻪵�ȥ��
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);
                      
����������������������������������������������������������������
____________________________________________________________
������ ����һ����ѵ�������ѵ�����������
fprintf('\nTraining Neural Network... \n')
%�����ʹ����Ϊ�����溯���Ļ������ã����õ�������
options = optimset('MaxIter', 400);
lambda = 0.5;%����Ĳ����ǿ��Ե����ģ�֮����ѧϰ���ߣ�ȷ����ε���
%����ȥ��һ���򵥵�J,��fmincgȥ�Ż�
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% ��nn_parameters��ȡ����������
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
�ڶ����� ���ӻ�������
 displayData(Theta1(:, 2:end));
 ��������Ԥ���������ȷ��
 pred = predict(Theta1, Theta2, X);
 mean(double(pred == y)
 %���ú���-��������������������������������������������������������
 function p = predict(Theta1, Theta2, X)%�����˲���������������ȷ�ʵļ���
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);%�����ǻ�����������
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');%�����Ƚ�����һ�����򴫲��õ���H2
[dummy, p] = max(h2, [], 2);%����������õ�������P,The max function might come in useful. In particular, the max function can also return the index of the max element,
% If your examples are in rows, then, you can use max(A, [], 2) to obtain the max for each row.
end