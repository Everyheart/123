clear ; close all ;clc
data= load('2222.txt');
X = data(:,2:11);
y =data(:,1);
m = length(y);


[X mu sigma] = featureNormalize(X);

X = [ones(m, 1) X];
alpha = 0.001;
iterations = 4000;
theta = zeros(11,1);%ע�⣬��Ϊ��theta0��Ҫ��һ����
%�����ά��J�Ĺ�ʽһ�����������������õ�GD������theta���󷨲�һ����������һ���������GD���������ڱ�ĵط���
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha,iterations);
%���ͼ�ǻ�
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%________________________________________
theta
price = theta' * [1;(([15.368373 3107.13 3169.08 3045.17 -30.14 -32.96	5.63 64.27 59.8 73.23] - mu)./sigma)']%������������ص�ָ�꣬���ܹ�Ԥ���Ӧ�õ�ֵ
