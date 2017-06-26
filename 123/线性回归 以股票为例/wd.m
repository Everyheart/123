clear ; close all ;clc

x = linspace(0,100,100);
y = linspace(0.100,100);
z = zeros(length(x),length(y));
for i = 1:length(x)
    for j = 1:length(y)
        z(i,j)= 1+2*(log(i))+10*(log(j));%这里的i的函数可以随便的改变
    end
end
z = z';
figure
surf(x,y,z)