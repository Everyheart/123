clear ; close all; clc
%————————————————————————————————————————
%第一部分引入数据，105*105的矩阵
A =load('3333.txt');
X= A(6:35,:);%最后5个数据是我们要的预测的。
size(X) ;           
y = (A(1,:)-A(5,:))./A(5,:)*100;
size(y);
m = length(y);
Y = cl(y,m);
y = Y;
X = X';
size(y)
size(X)
%测试集：可以修改的

Xval1 = A(11:40,:);
Xval = Xval1';
Yval1 = (A(6,:)-A(10,:))./A(10,:)*100;
s = length(Yval1);
yval = cl(Yval1,s) ;
%______________________________________________________________________
%扩大训练集：（处理高方差问题）
X1 = A(26:55,:);
Xv = X1';
Y1 = (A(21,:)-A(25,:))./A(25,:)*100;
s = length(Y1);
yv = cl(Y1,s) ;
X2 = A(56:85,:);
Xv2 = X2';
Y2 =  (A(51,:)-A(55,:))./A(55,:)*100;
s = length(Y2);
yv2 = cl(Y2,s) ;

X3= A(86:115,:);
Xv3 = X3';
Y3 =  (A(81,:)-A(85,:))./A(85,:)*100;
s = length(Y3);
yv3 = cl(Y2,s) ;
X = [X;Xv;Xv2;Xv3];
y = [y ; yv;yv2;yv3];
size(X)
size(y)
%_---------------------------------------------------------------------
%————————————————————————————————————————
%第二部分：设置输入的单元数量，输出的单元数量，隐含层的数量，
input_layer_size  = 30;  
hidden_layer_size = 90; 
num_labels = 9; 
%————————————————————————————————————————
%第三部分：随机产生初始的theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%————————————————————————————————————-———
%第四部分，使用正向传播：
lambda =1;
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
              J
pause;

%————————————————————————————————————————
%第四部分：进行检查
lambda = 1;
checkNNGradients(lambda);
pause;
% Also output the costFunction debugging values

%————————————————————————————————————————
%第五部分：进行训练，
options = optimset('MaxIter', 2000);
lambda = 0.001;%这里的参数是可以调整的，之后结合学习曲线，确定如何调整
checkNNGradients(lambda);
%这里去做一个简单的J,让fmincg去优化
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%————————————————————————————————————————
%第六部分，提取参数，计算出训练集的准确度：
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 pred = predict(Theta1, Theta2, X);
 P = mean(double(pred == y)) * 100;
 fprintf('Training Set Accuracy: \n');
 P
 fprintf('测试集训练准确率 \n');
 pred = predict (Theta1,Theta2,Xval);
 P =mean(double(pred == yval))*100
 %_______________________________________________________________
 %实验记录1：一万次迭代后的结果是  57.8853%
 %input_layer_size  = 100;  
%hidden_layer_size = 180; &几倍于输入的
%num_labels = 9; 
%options = optimset('MaxIter', 10000);
%lambda = 0.01;
%——————————————————————————————————
%实验记录2：1000次迭代后 43.8351%
%input_layer_size  = 100;  
%hidden_layer_size = 300; 
%num_labels = 9; 
%options = optimset('MaxIter', 1000);
%lambda = 0.01;%
%_________________________________
%第三次实验试验了5个小时，在14000次中断了，未知原因，还Hidden layer:300，
%大概到了1.5的cost，今晚可以睡觉的时候去做这个。
%——————————————————————————————
%第四次实验，迭代次数14000次，hidden layer size =300
%lambda = 0.01 ,cost到达了1.53。准确度达到了69.928%
%测试集的成功率只有26.63%，另外一组只有22%
%实际上表明了目前处于过度拟合的状态，因此需要想办法解决这个问题。也就是高方差，这个时候需要增加样本等方式解决