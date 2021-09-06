clc
clear all;
% close all;

%% Third Problem
C = [6 5];

f = C'*(-1);
A = [5 6; 5 2; 2 2; 3 .5; 7 8; 8 3];
b = [25 13 19 6 32 17];
lb = [0 0];

X = linprog(f,A,b,[],[],lb,[],[])

Z_star = C*X