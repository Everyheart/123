function [ theta,J_history] = gd( X,y,theta,alpha,iteratons)
J_history = zeros(iteratons,1);
m=length(y);
for iter=1:iteratons
    temp1 = theta(1)-alpha/m*sum(X*theta-y);
    temp2 = theta(2) - alpha/m*sum((X*theta-y).*X(:,2));%������õ�X�ĵڶ��У�Ҳ����X��ԭ�������ݣ�Xi
    theta(1)= temp1;
    theta(2)= temp2;
    J_history(iter) = costfunction(X,y,theta);
end
J_history;%������ʾ��������������ĵ�ֵ

end

