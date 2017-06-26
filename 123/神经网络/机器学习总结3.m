神经网络
第一部分：第一步，正向传播和预测（已知参数）
————————————————————————————————————————
input_layer_size  = 400;  
hidden_layer_size = 25; 
num_labels = 10;                   

load('ex3data1.mat');%这里的X，y在原来的数据中已经设置好，因此不用
m = size(X, 1);
sel = randperm(size(X, 1));%随机形成一个矩阵，将样本里的
sel = sel(1:100);
displayData(X(sel, :));%这个画图在上一个总结里
load('ex3weights.mat');%下载参数
pred = predict(Theta1, Theta2, X);%预测结果，这里用了整个X，因此我们可以用自己的数据去做。
mean(double(pred == y)) * 100%计算结果的准确率，比较每一个预测的和Y,最后均值
%调用函数------------------------------------------------------------------
function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
alpha1 = [ones(m,1) X];
alpha2 =sigmoid(Theta1*alpha1')';
m = size(alpha2,1);
alpha2 = [ ones(m,1) alpha2];
alpha3 = sigmoid(Theta2*alpha2');%求出最后一层下概率最大的那个，就是我们要找的数字，最后的用的max和那个没看懂，其实整体来说就是三层加上内容，特别注意有关的那个矩阵的构成
问题所在[y,p] = max(alpha3);%这里找到了最大的行，然后预测的结果就是我们要的，
p = p';
end
第二部分：使用反向传播法学习训练参数，最终进行预测
第一步：计算正向传播的误差函数结果
————————————————————————————————
load('ex4weights.mat');%导入参数值，下一步将它合并成一个新的举证
nn_params = [Theta1(:) ; Theta2(:)];
（1）对应函数调用的第一部分，计算出J，
lambda = 0;%设定初始的值
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
 （2）对应函数调用的第二部分：计算出正则化的J
 lambda = 1;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
 （插入的求偏导数的函数2） g = sigmoidGradient([1 -0.5 0 0.5 1]);也就是求一个这个函数的偏导数
 （初始化参数值，用函数3）
 initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
（这里将其再次合成一个矩阵）
initial_nn_params = [initial_Theta1(:) ; ini tial_Theta2(:)];
%调用函数1如下：
%————————————————————————————————————————
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)%调用了这么多的数，计算出前值
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));%将原来的参数恢复成原来的参数形式，因为函数应用的不含
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));%上面都是初始的值的设置
%——————————————————————————
（1）正向传播，计算出cost
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
n = size(a2,1);
a2 = [ones(n,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);%这是一个正向的传播，实现了求出我们要的a3
%下面一步是比较复杂的
for k = 1:num_labels%从1到第K个标签
重点y_k = (y == k);这里用y中的每一个位置的与K比较，如果一样，那么就是1 。
这里的重点是，y是给定的，给的就是0-10，那么这样的结果就是我们得到了对应的列向量，表示y-k也就是
第k个标签，对应的整体的列向量。
	a3_k = a3(:,k);%这里是得到了各个K的具体的值，这里只要第K列，找到了预测为k的样本
	J_K =1/m * sum(-y_k .* log(a3(:,k)) - (1 - y_k) .* log(1 - a3(:,k)));
    %计算求和出这时候的第K标签下的j
	J = J + J_K;%求和得到总的j
end
%——————————————————————————————
（2）正则化后面就是正则化的内容
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
%——————————————————————————————————————————
（3）求出对应的theta1和theta2 的偏导数
for t = 1:m%每个样本都要去做这个工作
	for k = 1:num_labels
		y_k = (y(t) == k);%判断第t个样本属于哪个标签
		delta3(k,1) = a3(t,k) - y_k;%直接减出这个误差，并且复制到这个矩阵里
    end
	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2(t,:)');
    %上面试用公式计算出第二个的所有的误差
    Theta1_grad = Theta1_grad + delta2 * a1(t,:);%这个就是反向传播求偏导数的一步
	Theta2_grad = Theta2_grad + delta3 * a2(t,:);%
end
% 下面求出了偏导数的具体J=0与不等于0的情况，带入公式，需要数学理解。
最后正则化修改后的公式是这样的
Theta1_grad(:,1) = 1/m .* Theta1_grad(:,1);
Theta1_grad(:,2:end) = 1/m .* Theta1_grad(:,2:end) + lambda/m .* Theta1(:,2:end);
Theta2_grad(:,1) = 1/m .* Theta2_grad(:,1);
Theta2_grad(:,2:end) = 1/m .* Theta2_grad(:,2:end) +lambda/m .* Theta2(:,2:end);
=========================================
（4）
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
%调用函数2——————————————————————————————————————
function g = sigmoidGradient(z)%求这个函数的偏导数
g = zeros(size(z));
g = sigmoid(z) .* (1 - sigmoid(z));
end
%调用函数3————————————————————————————————————
function W = randInitializeWeights(L_in, L_out)%这个对于所有的初始化都可以用
W = zeros(L_out, 1 + L_in);%初试化，下面是公式
epsilon_init= 0.12;%这个是题目给的，一般不用太大
W = rand(L_out,1 + L_in) * 2 * epsilon_init - epsilon_init;%实现随机形成一个阵，后用epsonlon处理，
end
——————————————————————————————————————————————————————————————
第二步：做偏导数正确性的分析，用两种不同的方法去求，公式和定义的方式去求，去验证
checkNNGradients;
%调用函数————————————————————————————————————
function checkNNGradients(lambda)
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
%创造了小的神经网络去做这个工作，也调用了两个函数，在具体用的时候再回来参考
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
第三步：正则化修改后
lambda = 3;
checkNNGradients(lambda);%再次检验后，用震泽华的去求
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);
                      
————————————————————————————————
____________________________________________________________
第三部 ：第一步：训练结果：训练出来结果：
fprintf('\nTraining Neural Network... \n')
%这里的使用是为了下面函数的基础设置，设置迭代次数
options = optimset('MaxIter', 400);
lambda = 0.5;%这里的参数是可以调整的，之后结合学习曲线，确定如何调整
%这里去做一个简单的J,让fmincg去优化
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% 从nn_parameters中取回两个参数
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
第二步： 可视化参数：
 displayData(Theta1(:, 2:end));
 第三步：预测整体的正确率
 pred = predict(Theta1, Theta2, X);
 mean(double(pred == y)
 %调用函数-————————————————————————————
 function p = predict(Theta1, Theta2, X)%给定了参数和样本进行正确率的计算
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);%以上是基本参数设置
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');%这里先进行了一次正向传播得到了H2
[dummy, p] = max(h2, [], 2);%这里用这个得到了最后的P,The max function might come in useful. In particular, the max function can also return the index of the max element,
% If your examples are in rows, then, you can use max(A, [], 2) to obtain the max for each row.
end