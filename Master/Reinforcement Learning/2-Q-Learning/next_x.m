function nx = next_x(x,phi,theta)
%NEXT_X Summary of this function goes here
%   Detailed explanation goes here
nx = x + cos(phi+theta) + sin(theta)*sin(phi);

end

