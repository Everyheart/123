clear;close all ;clc
%第一个画出成交量和指数的图，可以发现可以用K-mean发现异常点，数据的导出用Excel更加方便。
data = load('999999.xls');
X = data(:,4);
y = data(:,2);
plotdata(X,y);
m = length(y);

%______________________________________________________
%求出theta
X = [ones(m,1),X];
theta = zeros(2,1); %初始化2x1的矩阵
iterations=10000 ;%迭代1000次，这里要与alpha结合，alpha是学习速度，越高，学习速度更快，而迭代越小,而太大不好，太小也不好。
alpha = 0.01;
costfunction(X,y,theta)
theta = gd(X, y, theta, alpha, iterations);%这里的只是位置，调用函数，我们这里定义的迭代次数，对应里面的。
fprintf('%f %f \n',theta(1),theta(2))%这里显示两个参数要用这样的形式，第一个是格式，第二个是相关内容
hold on;
plot(X(:,2),X*theta,'-');%连接相关的图，这个图是X的第二列，因为第一列加了东西所以不用第一列
legend('Training data', 'Linear regression');%表明图中两个东西的名字，以区分
hold off %不再这个图加东西了
%________________________________________-
%可视化j,看)懂就行，需要的时候就调用
theta0_vals = linspace(1000, 5000, 100);%特别主义这里的图的区间需要调试，但是后面的这个100基本不用动，调试让最优点在中间
theta1_vals = linspace(-500, 500, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = costfunction(X, y, t);
    end
end

J_vals = J_vals';
figure;%画出3D的图
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');%注意这里的theta的表示方法

figure;%画出轮廓线
contour(theta0_vals, theta1_vals, J_vals, logspace(0,50,200))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%关于那个J和alpha，多元才能调试这个。


