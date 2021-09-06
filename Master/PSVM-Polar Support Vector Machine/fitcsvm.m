clc
clear all

rng(1); % For reproducibility
r = sqrt(rand(100,1)) % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

figure;
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
% ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;


m=2
    for i=1:200
        d_sq=0;
        for j=1:m%number of features
            d_sq=d_sq+data3(i,j)^2;%distance square
        end
        d=sqrt(d_sq);
        for j=1:m  
            data3(i,j)=acos(data3(i,j)/d);
        end
        data3(i,m+1)=d;

    end
    
data3=[data3 ones(2*100,1)];%for bias

m=length(data3(1,:))
n=length(theclass)

H=zeros(m,m)

for j=1:m-1
    H(j,j)=1
end

f = [];
for i=1:n
A(i,:)=data3(i,:).*-theclass(i);
b(i,1)=-1;%theclass(i)
end

opts = optimset('Algorithm','interior-point-convex','Display','off');
[x,fval,exitflag,output,lambda] = ...
quadprog(H,f,A,b,[],[],[],[],[],opts);

x

figure;
col=hsv(2);
for i=1:100
plot3(data3(i,1),data3(i,2),data3(i,3),'Or')
hold on
end
for i=101:200
plot3(data3(i,1),data3(i,2),data3(i,3),'ob')
hold on
end

x1=-2:.1:2;
d=-2:.1:2;
for i=1:length(d)
    y1(:,i)=(-x(1)*x1-x(2)*d(i) -x(4))/(x(3));
end
surf(x1,d,y1)