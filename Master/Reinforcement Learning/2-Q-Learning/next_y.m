function ny = next_y(y,phi,theta)
%NEXT_X Summary of this function goes here
%   Detailed explanation goes here
ny= y + sin(phi+theta) - sin(theta)*cos(phi);

end

