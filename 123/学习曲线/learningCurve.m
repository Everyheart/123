function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda,iter,input_layer_size,hidden_layer_size, num_labels) 
m = 200;
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
   initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

for i = 1:m%这里的M是X中样本的总数，要循环去一个个尝试训练出来我们想要的theta，然后再在测试的
	xset = X(1:i,1:100);%这个是从原
	yset = y(1:i);
    xset1 = Xval(1:i,1:100);
    yset1 = yval(1:i);
  options = optimset('MaxIter', iter);
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, xset, yset, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
      error_train(i) = cost(iter);
      costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, xset1, yset1, lambda);
   [nn_params1, cost1] = fmincg(costFunction, initial_nn_params, options);
	error_val(i) = cost1(iter);
end 
   

end
