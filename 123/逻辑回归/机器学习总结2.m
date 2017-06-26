��һ�����ݣ�
������������������������������������������������������������������������������������������������
������ͬ����ɢ��ͼ��ѧ������ҵ����ݣ���λ������Ȧ���Լ�������ɫ�ʹ�С�������α���XY��ͼ�������ݵ���ȡʡ����
function plotData(X, y)
figure; hold on;
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);)
%plot ��������������ɼ��������·ֱ�����ɵ���û����ɵģ������������ԡ�
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7)
hold off;
end
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
�ڶ������ݣ�
����������������������������������������������������������������������������������������������������
��������costfunction��ƫ�������õ����£�
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);
%���ú������¡�������������������������������������������������������
function [J, grad] = costFunction(theta, X, y)
m = length(y); 
J = 0;
grad = zeros(size(theta));
J=(1/m)*sum((-1.*y.*log(sigmoid(X*theta)))-(1.-y).*log(1.-sigmoid(X*theta)));
grad = (1/m).*X'*(sigmoid(X*theta)-y);
end
���������ݣ�
������������������������������������������������������������������������������������
�øߵȺ������Ż���������Ҿ��û���fminunc���������͹����ˣ�
% ʹ��fminunc���������ã�
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
%��������������������������fminunc�����ǲ���ȥд�ģ��Ǳ������ú���ϵͳ�ڵģ����ֻ��ȥ���úû������þ���
%��������������������������������������������������������������������������������
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
���ĸ����ݣ�Ԥ��һ�����ݵĸ��ʣ�Ҳ����һ�����ݣ���Ԥ���׼ȷ��
����������������������������������������������������������������������������������������
prob = sigmoid([1 45 85] * theta);
%���ú���������������������������������������������������������������������������������������
function g = sigmoid(z)
g = zeros(size(z));
g = 1./(1.+ exp(-1.*z));
end
%+����������������������������������������������������������������������������������������������
% ������ѵ������׼ȷ��
p = predict(theta, X);
mean(double(p == y)) * 100%���������p=y ����ȷ�ʣ�������ֵ��Ҳ����������ȷ��
%���ú���������������������������������������������������������������������������������������������������������������
function p = predict(theta, X)
m = size(X, 1); 
p = zeros(m, 1);

p = sigmoid(X*theta) >= 0.5;%�����ǽ����ڵ���0.5����P����������ȷ�ģ���1��ʾ��֮����ÿһ����Y�ȶԣ��ȵ�����ȷ��
end

�ڶ����ֵ�һ��
��������������������������������������������������������������������������������������������������
�������ı�������δ�������feature��Ҳ���Ƕ���ʽ�Ĵ��췽����
X = mapFeature(X(:,1), X(:,2));
initial_theta = zeros(size(X, 2), 1);%���������Ĳ������������ã�ʡ�������ã��ܷ���
%���ú�������������������������������������������������������
function out = mapFeature(X1, X2)
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);%���ｫ��1���6�ε����е����������������ɵĶ�������
    end
end
end
�ڶ������������򻯵�j
����������������������������������������������������������������
lambda = 1;%�������������Ҫ���Եģ�������Ҫ�ǿ����״������������������
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
%�����ˡ������������������������������������������������������򻯵ı�ﺯ��
function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); 
J = 0;
grad = zeros(size(theta));
J = (1/m)*sum((-1.*y.*log(sigmoid(X*theta)))-(1-y).*log(1.-sigmoid(X*theta)))+(lambda/(2*m))*sum(theta(2:end).^2);
grad = (1/ m).*X'*(sigmoid(X*theta)-y)+(lambda/m)*theta;
grad(1) = (1/m).*X(:,1)'*(sigmoid(X*theta)-y);
end
����������J��fmnuinc����
����������������������������������������������������������������������������
initial_theta = zeros(size(X, 2), 1);
�����Ƕ���fminunc�����ĳ�ʼ��������
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
%��������֮ǰ�ĺ��������һ�ͼ
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
���Ĳ���Ԥ�����ȷ�ʣ������õ�֮ǰ�ĺ�������������������������������������������
p = predict(theta, X);
�������֡���Ԫ�߼��ع�
��������������������������������������������������������������������������������
��Ԫ�߼��ع�õ�������Ҫ��theta��һԪ��ʱ��ֻ��һ������Ԫ��ʱ����������
���ȵĴ�costfunction���������򻯵�һ����
lambda = 0.1;num_labels = 10;   
[all_theta] = oneVsAll(X, y, num_labels, lambda);
%���ú�����������������������������������������������������������������������
%����ѭ����fmincg���������Ч�ĺ���ʵ�����ÿһ����Ӧ��theta����ٷ���һ�𣬾͵õ���������Ҫ������
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1);
n = size(X, 2);y 
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);%��ʼ���õ�theta��ʽ
options = optimset('GradObj', 'on', 'MaxIter', 50);%�����й�ѡ����ܽ��н������ĺ�����������Ҫ��ѭ������Ϊ�������ֻ����һ������Ĵ�theta��i���������ѭ��
for c = 1:num_labels,
     [theta] =  fmincg (@(t)(CostFunctionreg(t, X, (y == c), lambda)), initial_theta, options);
     all_theta(c,:) = theta;  %����c�е�thetaȫ����ɺ�����������Ľ�����ɹ���ѵ��������������Ҫ��theta
end;
end
�������ֵĵڶ�������������������������������������������������������������������
Ԥ�⣺����������ȥ����ϣ�ȥ������׼ȷ��
pred = predictOneVsAll(all_theta, X);
mean(double(pred == y)) * 100%���������ǲ�������ϵľ�ֵ��
%���õĺ���������������������������������������������������������������������������������
function p = predictOneVsAll(all_theta, X)
m = size(X, 1); %X�е������ĸ���
num_labels = size(all_theta, 1);%���е�theta���е�
p = zeros(size(X, 1), 1);%�������������Ļ�����P
X = [ones(m, 1) X];
[y,p] = max (sigmoid(all_theta*X'));%�ҵ����еĹ��Ƶ��У���ȷ�ĸ�����ߵ��Ǹ��ĸ��ʣ���������Ҫ�ĵ���
p = p';
end
�����Ϳ��Եõ����Ľ����ˡ�