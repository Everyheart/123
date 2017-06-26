function J = costfunction( X,y,theta )
m = size(X,1);
J=0;
m =size(X,1);
predictions= X*theta;
sqr = (predictions - y).^2;
J = 1/(2*m)*sum(sqr);



end

