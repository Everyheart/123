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
%��������������������������������������������������������������������������������
%�ڶ����֣���������ĵ�Ԫ����������ĵ�Ԫ�������������������
input_layer_size  = 100;  
hidden_layer1_size = 300; 
hidden_layer2_size = 300;
num_labels = 9; 
%��������������������������������������������������������������������������������
%�������֣����������ʼ��theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);initial_Theta3(:)];
%size(initial_nn_params):
%size(initial_Theta2):

%������������������������������������������������������������������������-������
%���Ĳ��֣�ʹ�����򴫲���
lambda =1;
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer1_size, ...
                   hidden_layer2_size,num_labels, X, y, lambda);
              J
pause;

%��������������������������������������������������������������������������������
%���Ĳ��֣����м��
%lambda = 1;
%checkNNGradients(lambda);

% Also output the costFunction debugging values

%��������������������������������������������������������������������������������
%���岿�֣�����ѵ����
options = optimset('MaxIter', 1000);
lambda = 1;%����Ĳ����ǿ��Ե����ģ�֮����ѧϰ���ߣ�ȷ����ε���
%checkNNGradients(lambda);
%����ȥ��һ���򵥵�J,��fmincgȥ�Ż�
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer1_size, ...
                   hidden_layer2_size,num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%��������������������������������������������������������������������������������
%�������֣���ȡ�����������ѵ������׼ȷ�ȣ�
nn = nn_params;in=input_layer_size;h1=hidden_layer1_size;h2=hidden_layer2_size;
ou = num_labels;
Theta1 =reshape(nn(1:(in+1)*h1),h1,(in+1));
Theta2 =reshape(nn(((in+1)*h1+1):(((in+1)*h1)+h1*(h2+1))),h2,(h1+1));
Theta3 =reshape(nn((1+(((in+1)*h1)+h1*(h2+1))):end),ou,(h2+1));
 pred = predict(Theta1, Theta2,Theta3 ,X);
 mean(double(pred == Y)) * 100
 cost
  
  