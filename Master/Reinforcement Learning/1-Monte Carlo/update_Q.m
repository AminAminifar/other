function nQ = update_Q( Q,x,phi,action,r,nx,nphi )
%UPDATE_Q Summary of this function goes here
%   Detailed explanation goes here
alpha = 0.5;
gamma = 0.9;
% nx
% nphi
[nx_i,nphi_i,~] = Q_index(nx,nphi,0);
[M,~] = max(Q(nx_i,nphi_i,:));

[x_i,phi_i,action_i] = Q_index(x,phi,action);

nQ = Q(x_i,phi_i,action_i) +...
    alpha*(r+gamma*M-Q(x_i,phi_i,action_i));

end

