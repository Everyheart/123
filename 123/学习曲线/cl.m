function Y = cl( y ,m)
Y=zeros(m,1);
for i=1:m
    if y(i)<-10
        Y(i)=1;
    elseif y(i)<-7
        Y(i)=2;
    elseif y(i)<-4
        Y(i)=3;
    elseif y(i)<-1
        Y(i) =4;
    elseif y(i)<1
        Y(i)=5;
    elseif y(i)<4
        Y(i)=6;
    elseif y(i)<7
        Y(i)=7;
    elseif y(i)<10
        Y(i)=8;
    else 
        Y(i)=9;
  
    end
end

  
    


end

