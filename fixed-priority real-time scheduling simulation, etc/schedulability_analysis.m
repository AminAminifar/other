function [ EDF_schedulable,RM_schedulable_guaranteed ] = schedulability_analysis(c,p,n)

%   Detailed explanation goes here
EDF_schedulable=logical(sum(c./p)<=1)
RM_schedulable_guaranteed=logical(sum(c./p)<=(n*((2^(1/n))-1)))

end

