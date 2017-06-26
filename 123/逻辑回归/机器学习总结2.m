第一个内容：
————————————————————————————————————————————————
画出不同分类散点图：学到如何找到数据，如何画出点和圈，以及设置颜色和大小，最后如何表明XY和图例，数据的提取省略了
function plotData(X, y)
figure; hold on;
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);)
%plot 后面跟的是两个成绩，而上下分别是完成的与没有完成的，而后面是属性。
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7)
hold off;
end
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
第二个内容：
——————————————————————————————————————————————————
求出最初的costfunction和偏导数，得到如下：
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);
%调用函数如下————————————————————————————
function [J, grad] = costFunction(theta, X, y)
m = length(y); 
J = 0;
grad = zeros(size(theta));
J=(1/m)*sum((-1.*y.*log(sigmoid(X*theta)))-(1.-y).*log(1.-sigmoid(X*theta)));
grad = (1/m).*X'*(sigmoid(X*theta)-y);
end
第三个内容：
——————————————————————————————————————————
用高等函数最优化结果：（我觉得会用fminunc函数基本就够用了）
% 使用fminunc基础的设置：
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
%这里调用了两个函数：（这里的fminunc函数是不用去写的，是本身设置好在系统内的，因此只用去设置好基础设置就行
%————————————————————————————————————————
function plotDecisionBoundary(theta, X, y)
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on
if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off
end
第四个内容，预测一组数据的概率，也就是一组数据，和预测的准确性
————————————————————————————————————————————
prob = sigmoid([1 45 85] * theta);
%调用函数——————————————————————————————————————————
function g = sigmoid(z)
g = zeros(size(z));
g = 1./(1.+ exp(-1.*z));
end
%+——）————————————————————————————————————————————
% 计算在训练集的准确性
p = predict(theta, X);
mean(double(p == y)) * 100%这个计算了p=y 的正确率，最后求均值，也就是最后的正确率
%调用函数——————————————————————————————————————————————————————
function p = predict(theta, X)
m = size(X, 1); 
p = zeros(m, 1);

p = sigmoid(X*theta) >= 0.5;%这里是将大于等于0.5的在P向量中是正确的，用1表示，之后用每一个和Y比对，等到的正确率
end

第二部分第一步
—————————————————————————————————————————————————
创造更多的变量，如何创造更多的feature，也就是多项式的创造方法，
X = mapFeature(X(:,1), X(:,2));
initial_theta = zeros(size(X, 2), 1);%这里吧最初的参数进行了设置，省的再设置，很烦。
%调用函数——————————————————————————
function out = mapFeature(X1, X2)
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);%这里将从1次项到6次的所有的由这两个变量构成的都构造了
    end
end
end
第二步：计算正则化的j
————————————————————————————————
lambda = 1;%这里的设置是需要调试的，调试主要是看你和状况，来看，后面讨论
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
%调用了——————————————————————————正则化的表达函数
function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); 
J = 0;
grad = zeros(size(theta));
J = (1/m)*sum((-1.*y.*log(sigmoid(X*theta)))-(1-y).*log(1.-sigmoid(X*theta)))+(lambda/(2*m))*sum(theta(2:end).^2);
grad = (1/ m).*X'*(sigmoid(X*theta)-y)+(lambda/m)*theta;
grad(1) = (1/m).*X(:,1)'*(sigmoid(X*theta)-y);
end
第三步，用J和fmnuinc函数
——————————————————————————————————————
initial_theta = zeros(size(X, 2), 1);
这里是对于fminunc函数的初始化的设置
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
%继续调用之前的函数，并且画图
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
第四步：预测出正确率，这里用的之前的函数————————————————————
p = predict(theta, X);
第三部分、多元逻辑回归
————————————————————————————————————————
多元逻辑回归得到我们想要的theta，一元的时候只有一个，多元的时候有许多个，
首先的从costfunction与上面正则化的一样，
lambda = 0.1;num_labels = 10;   
[all_theta] = oneVsAll(X, y, num_labels, lambda);
%调用函数——————————————————————————————————
%利用循环和fmincg这个更加有效的函数实现求出每一个对应的theta最后再放在一起，就得到了我们想要的内容
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1);
n = size(X, 2);y 
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);%初始设置的theta格式
options = optimset('GradObj', 'on', 'MaxIter', 50);%设置有关选项才能进行接下来的函数，并且需要用循环，因为这个函数只能求一个相隔的大theta（i），因此用循环
for c = 1:num_labels,
     [theta] =  fmincg (@(t)(CostFunctionreg(t, X, (y == c), lambda)), initial_theta, options);
     all_theta(c,:) = theta;  %将第c行的theta全部变成函数计算出来的结果，成功的训练出来了我们想要的theta
end;
end
第三部分的第二个————————————————————————————————
预测：这里用样本去做拟合，去看最后的准确率
pred = predictOneVsAll(all_theta, X);
mean(double(pred == y)) * 100%计算出最后是不是相符合的均值。
%调用的函数：——————————————————————————————————————
function p = predictOneVsAll(all_theta, X)
m = size(X, 1); %X中的样本的个数
num_labels = size(all_theta, 1);%所有的theta当中的
p = zeros(size(X, 1), 1);%构建样本个数的基础的P
X = [ones(m, 1) X];
[y,p] = max (sigmoid(all_theta*X'));%找到所有的估计当中，正确的概率最高的那个的概率，就是我们要的到的
p = p';
end
这样就可以得到最后的结论了。