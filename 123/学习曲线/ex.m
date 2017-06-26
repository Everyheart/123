clear ; close all; clc
%��������������������������������������������������������������������������������
%��һ�����������ݣ�105*105�ľ���
A =load('3333.txt');
X= A(6:105,:);%���5������������Ҫ��Ԥ��ġ�
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
%��������������������������������������������������������������������������������
%�ڶ����֣���������ĵ�Ԫ����������ĵ�Ԫ�������������������
input_layer_size  = 100;  
hidden_layer_size = 50; 
num_labels = 9; 
%��������������������������������������������������������������������������������
%�������֣����������ʼ��theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%������������������������������������������������������������������������-������
%���Ĳ��֣�ʹ�����򴫲���
lambda =1;
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
              J
pause;
%��������������������������������������������������������������������������������
%���Ĳ��֣����м��
lambda = 1;
checkNNGradients(lambda);
pause;
% Also output the costFunction debugging values

%��������������������������������������������������������������������������������
%���岿�֣�����ѵ����
options = optimset('MaxIter', 10);
lambda = 0.01;%����Ĳ����ǿ��Ե����ģ�֮����ѧϰ���ߣ�ȷ����ε���
checkNNGradients(lambda);
%����ȥ��һ���򵥵�J,��fmincgȥ�Ż�
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%��������������������������������������������������������������������������������
%�������֣���ȡ�����������ѵ������׼ȷ�ȣ�
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
 %ѧϰ����
 lambda = 0;iter = 10;%�ڼ���ѧϰ���ߵ�ʱ�����Ҫ��֤��һ�㡣
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda,iter,input_layer_size,hidden_layer_size, num_labels);%��һ�����Ƕ���X��cv���������������biasһ�н��д���
m=200
plot(1:m, error_train, 1:m, error_val);%����ֱ�ӾͰ���������ݻ�ͼ�ˣ�1��m�����ݣ�����������ֵ�����ڸ���������
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')