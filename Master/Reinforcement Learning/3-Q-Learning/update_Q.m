function nQ = update_Q( Q,x,y,g,d,action,r,nx,ny,ng,nd)
%UPDATE_Q Summary of this function goes here
%   Detailed explanation goes here
alpha = 0.1;
gamma = 0.5;


[M,~] = max(Q(nx,ny,ng,nd,:));

nQ = Q(x,y,g,d,action) +...
    alpha*(r+gamma*M-Q(x,y,g,d,action));

end

