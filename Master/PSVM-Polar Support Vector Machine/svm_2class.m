clc
clear all

% data(1,:)=[2 -1 1];%last column is for bias>always 1
% data(2,:)=[1 -1 1];
% target(1)=1;
% target(2)=-1;

number_class=2;
number_data_inClass=20;
in=data_f(number_class,number_data_inClass);
data(1:number_data_inClass,1:2)=in(1,:,:);
data(number_data_inClass+1:2*number_data_inClass,1:2)=in(2,:,:);
data=[data ones(2*number_data_inClass,1)];
target(1:number_data_inClass)=1;
target(number_data_inClass+1:2*number_data_inClass)=-1;

m=length(data(1,:))
n=length(target)

H=zeros(m,m)

for j=1:m-1
    H(j,j)=1
end

f = [];
for i=1:n
A(i,:)=data(i,:).*-target(i);
b(i,1)=-1;%target(i)
end


% H=[1 0;0 1]
% f=[];
% A=[1 1;2 2]
% b=[1;-1]

opts = optimset('Algorithm','trust-region-reflective ','Display','on');
[x,fval,exitflag,output,lambda] = ...
quadprog(H,f,A,b,[],[],[],[],[],opts);

x
fval
exitflag
output
lambda


x1=-10:.1:10;
y1=(-x(1)*x1 -x(3))/(x(2));
y2=(-x(1)*x1 -x(3)-1)/(x(2));
y3=(-x(1)*x1 -x(3)+1)/(x(2));
plot(x1,y1)
hold on
plot(x1,y2,'r')
hold on
plot(x1,y3,'r')
