function [ individual_out ] = mutation( individual_in )
%mutates an individuals.

individual_out = individual_in;
n = length(individual_in);
indices = randperm(n,2);
[individual_out(indices(1)), individual_out(indices(2))]=...
    deal(individual_in(indices(2)),individual_in(indices(1)));

end

