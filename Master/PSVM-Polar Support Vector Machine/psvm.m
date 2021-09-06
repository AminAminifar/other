function out = psvm(number_class,number_data_inClass)
%PSVM Summary of this function goes here
%   Detailed explanation goes here
% clc
% clear all
%  number_class=a;
%  number_data_inClass=b;
in=data_f(number_class,number_data_inClass);

m=length(in(1,1,:))
d_sq=0;%distance square

for class=1:number_class
%     data((class-1)*number_data_inClass+1:...
%         class*number_data_inClass,1:2)=in(class,:,:);
    
    for i=1:number_data_inClass
        d_sq=0;% ADD THINK
        for j=1:m%number of features
            d_sq=d_sq+in(class,i,j)^2;%distance square
        end
        d=sqrt(d_sq);
        for j=1:m  
            in(class,i,j)=acos(in(class,i,j)/d);
        end
        in(class,i,m+1)=d;
%         temp=zeros(1,m);
%         temp(1,:)=in(class,i,1:m);
%         temp=[temp d];
%         in(class,i,1:length(temp(1,:)))=temp(1,:);
    end
end
length(in(1,1,:))
out=in;
end

