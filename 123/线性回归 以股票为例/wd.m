clear ; close all ;clc

x = linspace(0,100,100);
y = linspace(0.100,100);
z = zeros(length(x),length(y));
for i = 1:length(x)
    for j = 1:length(y)
        z(i,j)= 1+2*(log(i))+10*(log(j));%�����i�ĺ����������ĸı�
    end
end
z = z';
figure
surf(x,y,z)