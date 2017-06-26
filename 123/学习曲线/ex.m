clear ; close all; clc
%————————————————————————————————————————
%第一部分引入数据，105*105的矩阵
A =load('3333.txt');
X= A(6:105,:);%最后5个数据是我们要的预测的。
size(X) ; 

y = (A(1,:)-A(5,:))./A(5,:)*100;
size(y);
m = length(y);
Y = cl(y,m);
y = Y;
X = X';
size(y);
size(X);
Xval1 = A(66:165,:);
Xval = Xval1';
Yval1 = (A(61,:)-A(65,:))./A(65,:)*100;
s = length(Yval1);
yval = cl(Yval1,s) ; 
size(Xval)
size(yval)
%————————————————————————————————————————
%第二部分：设置输入的单元数量，输出的单元数量，隐含层的数量，
input_layer_size  = 100;  
hidden_layer_size = 50; 
num_labels = 9; 
%————————————————————————————————————————
%第三部分：随机产生初始的theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%————————————————————————————————————-———
%第四部分，使用正向传播：
lambda =1;
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
              J
pause;
%————————————————————————————————————————
%第四部分：进行检查
lambda = 1;
checkNNGradients(lambda);
pause;
% Also output the costFunction debugging values

%————————————————————————————————————————
%第五部分：进行训练，
options = optimset('MaxIter', 10);
lambda = 0.01;%这里的参数是可以调整的，之后结合学习曲线，确定如何调整
checkNNGradients(lambda);
%这里去做一个简单的J,让fmincg去优化
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%————————————————————————————————————————
%第六部分，提取参数，计算出训练集的准确度：
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 pred = predict(Theta1, Theta2, X);
 P = mean(double(pred == Y)) * 100;
 fprintf('Accuracy: \n');
 P
 pause
 %____________________________________________________________________
 %学习曲线
 lambda = 0;iter = 10;%在计算学习曲线的时候必须要保证这一点。
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda,iter,input_layer_size,hidden_layer_size, num_labels);%这一步就是对于X和cv集合里的样本加上bias一列进行处理
m=200
plot(1:m, error_train, 1:m, error_val);%这里直接就把里面的数据画图了，1到m个数据，两个向量的值，对于各个样本的
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')