clear ; close all; clc
%————————————————————————————————————————
%第一部分引入数据，105*105的矩阵
A =load('3333.txt');
X= A(6:105,:);%最后5个数据是我们要的预测的。
size(X) ;           
y = (A(1,:)-A(5,:))./A(5,:)*100;
size(y);
m = length(y);
Y = cl(y,m);
y = Y;
X = X';
size(y);
size(X);
%————————————————————————————————————————
%第二部分：设置输入的单元数量，输出的单元数量，隐含层的数量，
input_layer_size  = 100;  
hidden_layer1_size = 300; 
hidden_layer2_size = 300;
num_labels = 9; 
%————————————————————————————————————————
%第三部分：随机产生初始的theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:);initial_Theta3(:)];
%size(initial_nn_params):
%size(initial_Theta2):

%————————————————————————————————————-———
%第四部分，使用正向传播：
lambda =1;
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer1_size, ...
                   hidden_layer2_size,num_labels, X, y, lambda);
              J
pause;

%————————————————————————————————————————
%第四部分：进行检查
%lambda = 1;
%checkNNGradients(lambda);

% Also output the costFunction debugging values

%————————————————————————————————————————
%第五部分：进行训练，
options = optimset('MaxIter', 1000);
lambda = 1;%这里的参数是可以调整的，之后结合学习曲线，确定如何调整
%checkNNGradients(lambda);
%这里去做一个简单的J,让fmincg去优化
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer1_size, ...
                   hidden_layer2_size,num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%————————————————————————————————————————
%第六部分，提取参数，计算出训练集的准确度：
nn = nn_params;in=input_layer_size;h1=hidden_layer1_size;h2=hidden_layer2_size;
ou = num_labels;
Theta1 =reshape(nn(1:(in+1)*h1),h1,(in+1));
Theta2 =reshape(nn(((in+1)*h1+1):(((in+1)*h1)+h1*(h2+1))),h2,(h1+1));
Theta3 =reshape(nn((1+(((in+1)*h1)+h1*(h2+1))):end),ou,(h2+1));
 pred = predict(Theta1, Theta2,Theta3 ,X);
 mean(double(pred == Y)) * 100
 cost
  
  