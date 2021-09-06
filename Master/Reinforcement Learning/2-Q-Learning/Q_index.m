function [x_index,phi_index,action_index] = Q_index( x,phi,action )
%Q Summary of this function goes here
%   Detailed explanation goes here
a = 0;
b = 8;
b2 = 9;
b3 = 9.7;
d = 10;
c3 = 10.3;
c2 = 11;
c = 12;
d = 20;
density1 = 1;
density2 = 10;
density3 = 100;
density4 = 1000;
if x>=a && x<b
    x_index = floor(x*density1);
elseif x>=b && x<b2
    x = x - b;
    x_index = b*density1 + floor(x*density2);
elseif x>=b2 && x<b3
    x = x - b2;
    x_index = b*density1 + (b2-b)*density2 + floor(x*density3);
% elseif x>=b3 && x<d
%     x = x - b3;
%     x_index = b*density1 + (b2-b)*density2 +...
%         (b3-b2)*density3  + floor(x*density4);
% elseif x<a
%     x_index = 1;
% elseif x>=d
%     x_index = b*density1 + (b2-b)*density2 +...
%         (b3-b2)*density3  + (d-b3)*density4 ;
% end

elseif x>=b3 && x<c3
    x = x - b3;
    x_index = b*density1 + (b2-b)*density2 +...
        (b3-b2)*density3  + floor(x*density4);
elseif x>=c3 && x<c2
    x = x - c3;
    x_index = b*density1 + (b2-b)*density2 +...
        (b3-b2)*density3  + (c3-b3)*density4  + floor(x*density3);
elseif x>=c2 && x<c
    x = x - c2;
    x_index = b*density1 + (b2-b)*density2 +...
        (b3-b2)*density3  + (c3-b3)*density4 +...
        (c2-c3)*density3  + floor(x*density2);
elseif x>=c && x<d
    x = x - c;
    x_index = b*density1 + (b2-b)*density2 +...
        (b3-b2)*density3  + (c3-b3)*density4 +...
        (c2-c3)*density3  + (c-c2)*density2  + floor(x*density1);

elseif x<a
    x_index = 1;
elseif x>=d
    x = x - c;
    x_index = b*density1 + (b2-b)*density2 +...
        (b3-b2)*density3  + (c3-b3)*density4 +...
        (c2-c3)*density3  + (c-c2)*density2  + (d-c)*density1;
end

% x_index
% floor(x_index) + 1
x_index = round(x_index) + 1;
% a = 0;
% b = 80;
% c = 110;
% d = ceil(pi);
% density1 = 1;
% density2 = 5;

% if phi>=a && phi<b
%     phi_index = 1 + floor(phi*density1);
% elseif phi>=b && phi<c
%     phi = phi - b;
%     phi_index = b*density1 + floor(phi*density2);
% elseif phi>=c && phi<d
%     phi = phi - c;
%     phi_index = b*density1 + (c-b)*density2 + floor(phi*density1);
% elseif x<a
%     phi_index = 1;
% elseif x>=d;
%     phi_index =  b*density1 + (c-b)*density2 + (d-c)*density1;
% end
phi_index = round(1 + floor(phi));
action_index = round(1 + floor(action));
% if action < 5 && 0<=action
%     action_index = round(1 + floor(action));
% else
%     action_index = round(10 + action);
% end
% action_index = round(1 + floor(action));

% action_value = Q_matrix(x_index,phi_index,action_index);

end

