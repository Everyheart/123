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
size(y)
size(X)
%��������������������������������������������������������������������������������
%�ڶ����֣���������ĵ�Ԫ����������ĵ�Ԫ�������������������
input_layer_size  = 100;  
hidden_layer_size = 300; 
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
options = optimset('MaxIter', 14000);
lambda = 0.001;%����Ĳ����ǿ��Ե����ģ�֮����ѧϰ���ߣ�ȷ����ε���
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
 %_______________________________________________________________
 %ʵ���¼1��һ��ε�����Ľ����  57.8853%
 %input_layer_size  = 100;  
%hidden_layer_size = 180; &�����������
%num_labels = 9; 
%options = optimset('MaxIter', 10000);
%lambda = 0.01;
%��������������������������������������������������������������������
%ʵ���¼2��1000�ε����� 43.8351%
%input_layer_size  = 100;  
%hidden_layer_size = 300; 
%num_labels = 9; 
%options = optimset('MaxIter', 1000);
%lambda = 0.01;%
%_________________________________
%������ʵ��������5��Сʱ����14000���ж��ˣ�δ֪ԭ�򣬻�Hidden layer:300��
%��ŵ���1.5��cost���������˯����ʱ��ȥ�������
%������������������������������������������������������������
%���Ĵ�ʵ�飬��������14000�Σ�hidden layer size =300
%lambda = 0.01 ,cost������1.53��׼ȷ�ȴﵽ��69.928%
%���Լ��ĳɹ���ֻ��26.63%������һ��ֻ��22%