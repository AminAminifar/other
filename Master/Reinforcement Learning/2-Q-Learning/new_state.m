function [ nx,ny,nphi ] = new_state( x,y,phi,theta )
%NEW_STATE Summary of this function goes here
%   Detailed explanation goes here
nx = x + cosd(phi+theta) + sind(theta)*sind(phi);
if nx<0
    nx = 0;
elseif nx>20
    nx = 20;
end
ny= y + sind(phi+theta) - sind(theta)*cosd(phi);
b = 4;
nphi = phi - asind((2*sin(theta)/b));
nphi = mod(nphi,180);
% if nphi<-180
%     nphi = nphi + 360;
% elseif nphi<0
%     nphi = nphi + 180;
%  elseif nphi>360
%     nphi = nphi - 360;
%    
% elseif nphi>360
%     nphi = nphi - 360;
% end
end

