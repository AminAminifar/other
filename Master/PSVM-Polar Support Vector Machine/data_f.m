function  out = data_f( number_class,number_data_inClass )
%DATA Summary of this function goes here
%generates data

%   Detailed explanation goes here

sign=[-1 1];
distances=3:5;

for i=1:number_class
    center_x=sign(randi(2))*...
        i*distances(randi(length(distances)));
    center_y=sign(randi(2))*...
        i*distances(randi(length(distances)));
    r=1;
for j=1:number_data_inClass
n(i,j,:)= normrnd([center_x center_y],r);
end
end
out=n;

f=figure('Name','normal distribution circle')
col=hsv(10);
for i=1:number_class
plot(n(i,:,1),n(i,:,2),'o','color',col(i,:))
hold on
end
% hold off
end

