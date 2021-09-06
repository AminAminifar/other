clc
clear all

% data(1,:)=[2 -1 1];%last column is for bias>always 1
% data(2,:)=[1 -1 1];
% target(1)=1;
% target(2)=-1;

number_class=3;
number_data_inClass=200;
in=psvm(number_class,number_data_inClass);

m1=length(in(1,1,:))
%     f=figure('Name','22normal distribution circle');
%     plot(0,0)
%     hold on
for class=1:number_class
    data(1:number_data_inClass,1:m1)=in(class,:,:);
    o=1;
    for other=1:number_class
        if other~=class
            data(o*number_data_inClass+1:...
                (o+1)*number_data_inClass,1:m1)=in(other,:,:);
            o=o+1;
        end
    end
    if class==1%adding one for bias "just one time"
        data=[data ones(number_class*number_data_inClass,1)];
    end
    target(1:number_data_inClass)=1;
    target(number_data_inClass+1:number_class*number_data_inClass)=-1;
    
    m=length(data(1,:))
    n=length(target)
    
    H=zeros(m,m);
    
    for j=1:m-1
        H(j,j)=1;
    end
    
    f = [];
    A(1:n,1:m)=zeros(n,m);
    for i=1:n
        A(i,:)=data(i,:).*-target(i);
        b(i,1)=-1;%target(i)
    end
    
    
    % H=[1 0;0 1]
    % f=[];
    % A=[1 1;2 2]
    % b=[1;-1]
    
    opts = optimset('Algorithm','interior-point-convex','Display','off');
    [x,fval,exitflag,output,lambda] = ...
        quadprog(H,f,A,b,[],[],[],[],[],opts);
    
    x
    % fval
    % exitflag
    % output
    % lambda
    
%     x1=-1:.1:1;
%     d=-1:.1:1;
%     for i=1:length(d)
%     y1(:,i)=(-x(1)*x1-x(2)*d(i) -x(4))/(x(3));
%     end
%     y2=(-x(1)*x1-x(2)*d -x(4)-1)/(x(3));
%     y3=(-x(1)*x1-x(2)*d -x(4)+1)/(x(3));
%     surf(x1,d,y1)
%     hold on
    % plot(x1,y2,'r')
    % hold on
    % plot(x1,y3,'r')
    
    pause(2)
end

f=figure('Name','22normal distribution circle')
col=hsv(number_class);
for i=1:number_class
plot(in(i,:,1),in(i,:,2),'o','color',col(i,:))
hold on
end
    