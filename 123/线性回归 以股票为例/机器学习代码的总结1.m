%初始化（原本）
clear ; close all ;clc
    %建立新的文件，函数用来调用，函数如下
    function A = warmUpExercise()%函数的前面的内容function,表示我们要用的内容，而A是我们在副本1中要表示出来的内容，而后面的warmU...是我们在原本中调用时的表示
    %对于函数的初始化：A是矩阵
    A= [];
    A=eye(5) %A是5阶单位矩阵
    end %完成函数定义的循环，得到我们要的，提取用warmUp...就行
%原本的第一个调用函数：
fprintf:('程序显示的内容 \n');%这里主要对调用的函数进行描述，\n表示回车
warmUpExercise()
fprintf:('操作提示 \n');
pause;
%_____________________________________________________________________________________________________
data = load('ex1data1.txt');%载入数据1,这里的数据是两列，最后载入后得到的是两列的矩阵
x = data(:,1) ; y=data(:,2); %分别提取第一列和第二列，注意写法
plotData(X,y)这里是画图，注意这里调用了plotData函数，如下
     function plotData(x,y) 这里没有等号，因为我们直接写了这个，相当于等号左侧
      figure %首先创建一个这样的图  
      plot(x, y, 'rx', 'MarkerSize', 10);  画图需要对图中的参数进行设置，主要包括,xy两个变量，rx表示红色的×，而最后的两项表示图像的大小的设定,注意这里的两个rx,和大小的设定要加引号
      ylabel('Profit in $10,000s');
     xlabel('Population of City in 10,000s'); 
     end  这里必须要调用函数，否则是没法画图的，已经尝试
 %_____________________________________________________________线性回归的求解
X = [ones(m, 1), data(:,1)]; %加一行到原来的数据中，注意加第一行，因为这样对应的就是theta0的
theta = zeros(2, 1);  %设置初始的
iterations = 1500;   
alpha = 0.01;%alpha的选择通常很难，可以考虑）0.01.0.1,1,0.001等，对于迭代次数也是可以如此，而将之画图，是判断优化程度的最佳工具。
     function J = computeCost(X, y, theta)右边是调用的，左边的是我们求的
     初始几个值：
     m=length(y);%找到有多少个，因为是单列的，所以用这个，如果是矩阵用size来写
     J=0;
     m =size(X,1);%重复的表达了，同上。
     predictions= X*theta;
     sqr = (predictions - y).^2;
     J = 1/(2*m)*sum(sqr);
     这里的总结起来就是： J= 1/(2*m)*sum((X*theta-y).^2);
     end
computeCost(X, y, theta)  %调用函数，计算出J
下面开始计算我们要的参数去拟合：
theta = gradientDescent(X, y, theta, alpha, iterations);
调用函数如下：
     function [theta, J_history ] = gradientDescent(X, y, theta, alpha, num_iters) %这里的重点是迭代需要的几个参数，要把握   
     m = length(y); 
     J_history = zeros(num_iters, 1); 
     for iter = 1:num_iters;
         temp1 = theta(1)-alpha/m*sum(X*theta - y);%这里没有theta0,只能用1
         temp2 = theta(2)-alpha/m*sum((X*theta -y) .*X(:,2));
         theta(1) = temp1;
         theta(2) = temp2;
         J_history(iter) = computeCost(X, y, theta); %这里调用了上面一个函数，实现对J的计算，这里的计算结果储存到了J_history中。
     end
     end
     
fprintf('%f %f \n', theta(1), theta(2));%输出两个值
hold on; %保持原来的图依然可以看见，再新加
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure   
————————————————————————————————————————————————————————————————————————————————
可视化我们的J，
theta0_vals = linspace(-10, 10, 100);%设置画图的空间，-10到10 之间的间距用1/100
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));%初始化J-val,形成100*100的0阵
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)%两个1-100的循环
	  t = [theta0_vals(i); theta1_vals(j)];    %把每一个的theta赋予t上再去计算出J，
	  J_vals(i,j) = computeCost(X, y, t);
    end
end
J_vals = J_vals';%转过来，
figure;
surf(theta0_vals, theta1_vals, J_vals)%画3D图用的，用着3个维度，画出3d的图
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))%画出轮廓线
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
接下来就是画出J和迭代次数相关的，对于alpha的图，但是这里得有J_history,
%必须要有[theta, J_history] = gradientDescentMulti(X, y, theta, alpha,iterations);这一步
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');