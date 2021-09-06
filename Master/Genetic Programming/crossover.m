function [ individual_out1, individual_out2 ] = crossover( individual_in1, individual_in2 )
%CROSSOVER Summary of this function goes here
%   Detailed explanation goes here
% individual_out1 = individual_in1;
% individual_out2 = individual_in2;
index1 = individual_in1.depthfirstiterator;
index2 = individual_in2.depthfirstiterator;
node1 = 1;
node2 = 1;
while node1==1
node1 = index1(randi(length(index1)));
end
while node2==1
node2 = index2(randi(length(index2)));
end
parent1 = individual_in1.getparent(node1);
parent2 = individual_in2.getparent(node2);
temp1 = individual_in1.chop(node1);
temp2 = individual_in2.chop(node2);
subtree1 = individual_in1.subtree(node1);
subtree2 = individual_in2.subtree(node2);
individual_out1 = temp1.graft(parent1, subtree2);
individual_out2 = temp2.graft(parent2, subtree1);

% disp(individual_in1.tostring)
% disp(individual_in2.tostring)
% disp('temp:')
% disp(temp1.tostring)
% disp(temp2.tostring)
% disp('subtree:')
% disp(subtree1.tostring)
% disp(subtree2.tostring)
% disp('individual_out:')
% disp(individual_out1.tostring)
% disp(individual_out2.tostring)

end

