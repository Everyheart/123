clear ; close all ;clc
data= load('2222.txt');
X = data(:,2:11);
y =data(:,1);
m = length(y);


[X mu sigma] = featureNormalize(X);

X = [ones(m, 1) X];
alpha = 0.001;
iterations = 4000;
theta = zeros(11,1);%注意，因为有theta0，要加一个。
%计算多维的J的公式一样，区别在于我们用的GD有区别，theta的求法不一样，其他都一样，这里的GD，可以用于别的地方了
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha,iterations);
%这个图是画
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%________________________________________
theta
price = theta' * [1;(([15.368373 3107.13 3169.08 3045.17 -30.14 -32.96	5.63 64.27 59.8 73.23] - mu)./sigma)']%在里面输入相关的指标，就能够预测出应该的值
