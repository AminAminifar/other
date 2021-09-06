clc
clear all;
% close all;

%% Second Problem
C = [3 4];

f = C'*(-1);
A = [3 1; 2 -3];
b = [1 1];%[1 2 4 7]&[1 3 5 6]
lb = [0 0];

X0 = linprog(f,A,b,[],[],lb,[],[])

Z0 = C*X0
%% 
b = [6 7];%[1 2 4 7]&[1 3 5 6]
X1 = linprog(f,A,b,[],[],lb,[],[])

Z1 = C*X1
%% 
C_ = [0 0 1];
t = [5 6];

f = C_'*(-1);
A = [3 1 t(1); 2 -3 t(2);-1*C(1) -1*C(2) Z1-Z0];
b = [6 7 -Z0];%[1 2 4 7]&[1 3 5 6]
lb = [0 0 0];
ub = [inf inf 1];

X = linprog(f,A,b,[],[],lb,ub,[])

Z_star = C*X(1:2)