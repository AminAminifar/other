function [ indv_index1, indv_index2 ] = selection( pop )
%SELECTION: select pairs for recombination from population 

n = length(pop);
rand = randperm(n,5);
%% 
indv_fitness = [];
for i=1:5
    indv_fitness(i) = pop(rand(i)).fitness;
end
[~,I] = sort(indv_fitness);
[ indv_index1, indv_index2 ] = deal(rand(I(4)),rand(I(5)));
end

