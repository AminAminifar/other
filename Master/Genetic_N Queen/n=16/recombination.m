function [ individual_out1, individual_out2 ] = recombination( individual_in1, individual_in2 )
%RECOMBINATION: gets 2 individuals and produce 2 children from them.
[ individual_out1, individual_out2 ] = deal(individual_in1, individual_in2);
n = length(individual_in1);
%%
crossover_point = randi(n-1);
%%
temp1 = [setdiff(individual_in2(crossover_point+1:n), individual_in1(1:crossover_point)),...
    setdiff(individual_in2(1:crossover_point), individual_in1(1:crossover_point))];
temp2 = [setdiff(individual_in1(crossover_point+1:n), individual_in2(1:crossover_point)),...
    setdiff(individual_in1(1:crossover_point), individual_in2(1:crossover_point))];


% if(~isempty(temp1))
% [ individual_out1, individual_out2 ] = ...
%     deal([individual_in1(1:crossover_point),temp1(1:n-crossover_point)],...
%     [individual_in2(1:crossover_point),temp2(1:n-crossover_point)]);
% end


if(~isempty(temp1))
individual_out1 = ...
    [individual_in1(1:crossover_point),temp1(1:n-crossover_point)];

end
if(~isempty(temp2))
individual_out2 = ...
    [individual_in2(1:crossover_point),temp2(1:n-crossover_point)];
end



end

