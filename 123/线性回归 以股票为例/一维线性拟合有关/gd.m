function [ theta,J_history] = gd( X,y,theta,alpha,iteratons)
J_history = zeros(iteratons,1);
m=length(y);
for iter=1:iteratons
    temp1 = theta(1)-alpha/m*sum(X*theta-y);
    temp2 = theta(2) - alpha/m*sum((X*theta-y).*X(:,2));%这里的用的X的第二列，也就是X的原本的数据，Xi
    theta(1)= temp1;
    theta(2)= temp2;
    J_history(iter) = costfunction(X,y,theta);
end
J_history;%用于显示这个函数，看最后的的值

end

