%��ʼ����ԭ����
clear ; close all ;clc
    %�����µ��ļ��������������ã���������
    function A = warmUpExercise()%������ǰ�������function,��ʾ����Ҫ�õ����ݣ���A�������ڸ���1��Ҫ��ʾ���������ݣ��������warmU...��������ԭ���е���ʱ�ı�ʾ
    %���ں����ĳ�ʼ����A�Ǿ���
    A= [];
    A=eye(5) %A��5�׵�λ����
    end %��ɺ��������ѭ�����õ�����Ҫ�ģ���ȡ��warmUp...����
%ԭ���ĵ�һ�����ú�����
fprintf:('������ʾ������ \n');%������Ҫ�Ե��õĺ�������������\n��ʾ�س�
warmUpExercise()
fprintf:('������ʾ \n');
pause;
%_____________________________________________________________________________________________________
data = load('ex1data1.txt');%��������1,��������������У���������õ��������еľ���
x = data(:,1) ; y=data(:,2); %�ֱ���ȡ��һ�к͵ڶ��У�ע��д��
plotData(X,y)�����ǻ�ͼ��ע�����������plotData����������
     function plotData(x,y) ����û�еȺţ���Ϊ����ֱ��д��������൱�ڵȺ����
      figure %���ȴ���һ��������ͼ  
      plot(x, y, 'rx', 'MarkerSize', 10);  ��ͼ��Ҫ��ͼ�еĲ����������ã���Ҫ����,xy����������rx��ʾ��ɫ�ġ��������������ʾͼ��Ĵ�С���趨,ע�����������rx,�ʹ�С���趨Ҫ������
      ylabel('Profit in $10,000s');
     xlabel('Population of City in 10,000s'); 
     end  �������Ҫ���ú�����������û����ͼ�ģ��Ѿ�����
 %_____________________________________________________________���Իع�����
X = [ones(m, 1), data(:,1)]; %��һ�е�ԭ���������У�ע��ӵ�һ�У���Ϊ������Ӧ�ľ���theta0��
theta = zeros(2, 1);  %���ó�ʼ��
iterations = 1500;   
alpha = 0.01;%alpha��ѡ��ͨ�����ѣ����Կ��ǣ�0.01.0.1,1,0.001�ȣ����ڵ�������Ҳ�ǿ�����ˣ�����֮��ͼ�����ж��Ż��̶ȵ���ѹ��ߡ�
     function J = computeCost(X, y, theta)�ұ��ǵ��õģ���ߵ����������
     ��ʼ����ֵ��
     m=length(y);%�ҵ��ж��ٸ�����Ϊ�ǵ��еģ����������������Ǿ�����size��д
     J=0;
     m =size(X,1);%�ظ��ı���ˣ�ͬ�ϡ�
     predictions= X*theta;
     sqr = (predictions - y).^2;
     J = 1/(2*m)*sum(sqr);
     ������ܽ��������ǣ� J= 1/(2*m)*sum((X*theta-y).^2);
     end
computeCost(X, y, theta)  %���ú����������J
���濪ʼ��������Ҫ�Ĳ���ȥ��ϣ�
theta = gradientDescent(X, y, theta, alpha, iterations);
���ú������£�
     function [theta, J_history ] = gradientDescent(X, y, theta, alpha, num_iters) %������ص��ǵ�����Ҫ�ļ���������Ҫ����   
     m = length(y); 
     J_history = zeros(num_iters, 1); 
     for iter = 1:num_iters;
         temp1 = theta(1)-alpha/m*sum(X*theta - y);%����û��theta0,ֻ����1
         temp2 = theta(2)-alpha/m*sum((X*theta -y) .*X(:,2));
         theta(1) = temp1;
         theta(2) = temp2;
         J_history(iter) = computeCost(X, y, theta); %�������������һ��������ʵ�ֶ�J�ļ��㣬����ļ��������浽��J_history�С�
     end
     end
     
fprintf('%f %f \n', theta(1), theta(2));%�������ֵ
hold on; %����ԭ����ͼ��Ȼ���Կ��������¼�
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure   
����������������������������������������������������������������������������������������������������������������������������������������������������������������
���ӻ����ǵ�J��
theta0_vals = linspace(-10, 10, 100);%���û�ͼ�Ŀռ䣬-10��10 ֮��ļ����1/100
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));%��ʼ��J-val,�γ�100*100��0��
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)%����1-100��ѭ��
	  t = [theta0_vals(i); theta1_vals(j)];    %��ÿһ����theta����t����ȥ�����J��
	  J_vals(i,j) = computeCost(X, y, t);
    end
end
J_vals = J_vals';%ת������
figure;
surf(theta0_vals, theta1_vals, J_vals)%��3Dͼ�õģ�����3��ά�ȣ�����3d��ͼ
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))%����������
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
���������ǻ���J�͵���������صģ�����alpha��ͼ�������������J_history,
%����Ҫ��[theta, J_history] = gradientDescentMulti(X, y, theta, alpha,iterations);��һ��
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');