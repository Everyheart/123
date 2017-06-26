clear;close all ;clc
%��һ�������ɽ�����ָ����ͼ�����Է��ֿ�����K-mean�����쳣�㣬���ݵĵ�����Excel���ӷ��㡣
data = load('999999.xls');
X = data(:,4);
y = data(:,2);
plotdata(X,y);
m = length(y);

%______________________________________________________
%���theta
X = [ones(m,1),X];
theta = zeros(2,1); %��ʼ��2x1�ľ���
iterations=10000 ;%����1000�Σ�����Ҫ��alpha��ϣ�alpha��ѧϰ�ٶȣ�Խ�ߣ�ѧϰ�ٶȸ��죬������ԽС,��̫�󲻺ã�̫СҲ���á�
alpha = 0.01;
costfunction(X,y,theta)
theta = gd(X, y, theta, alpha, iterations);%�����ֻ��λ�ã����ú������������ﶨ��ĵ�����������Ӧ����ġ�
fprintf('%f %f \n',theta(1),theta(2))%������ʾ��������Ҫ����������ʽ����һ���Ǹ�ʽ���ڶ������������
hold on;
plot(X(:,2),X*theta,'-');%������ص�ͼ�����ͼ��X�ĵڶ��У���Ϊ��һ�м��˶������Բ��õ�һ��
legend('Training data', 'Linear regression');%����ͼ���������������֣�������
hold off %�������ͼ�Ӷ�����
%________________________________________-
%���ӻ�j,��)�����У���Ҫ��ʱ��͵���
theta0_vals = linspace(1000, 5000, 100);%�ر����������ͼ��������Ҫ���ԣ����Ǻ�������100�������ö������������ŵ����м�
theta1_vals = linspace(-500, 500, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = costfunction(X, y, t);
    end
end

J_vals = J_vals';
figure;%����3D��ͼ
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');%ע�������theta�ı�ʾ����

figure;%����������
contour(theta0_vals, theta1_vals, J_vals, logspace(0,50,200))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%�����Ǹ�J��alpha����Ԫ���ܵ��������


