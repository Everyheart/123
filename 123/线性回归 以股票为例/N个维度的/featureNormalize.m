function [ X_norm,mu,sigma] = featureNormalize( X )
%将X正则化，这个公式可以复制，直接去调用就行。
X_norm = X;
mu = zeros(1,size(X,2));
sigma = zeros(1,size(X,2));
mu  = mean(X);
sigma =std(X);%这个就是标准差，X-u/标准差
X_norm = (X-ones(size(X,1),1)*mu)./(ones(size(X,1),1)*sigma);%公式
end

