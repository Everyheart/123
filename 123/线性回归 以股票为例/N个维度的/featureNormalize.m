function [ X_norm,mu,sigma] = featureNormalize( X )
%��X���򻯣������ʽ���Ը��ƣ�ֱ��ȥ���þ��С�
X_norm = X;
mu = zeros(1,size(X,2));
sigma = zeros(1,size(X,2));
mu  = mean(X);
sigma =std(X);%������Ǳ�׼�X-u/��׼��
X_norm = (X-ones(size(X,1),1)*mu)./(ones(size(X,1),1)*sigma);%��ʽ
end

