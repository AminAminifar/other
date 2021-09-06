function nphi = next_phi( phi,theta )
%NEXT_PHI Summary of this function goes here
%   Detailed explanation goes here
b = 4;
nphi = phi - sinh((2*sin(theta)/b));

end

