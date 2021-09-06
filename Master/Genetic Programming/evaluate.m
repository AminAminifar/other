function [ out ] = evaluate( individual_org )
%EVALUATE Summary of this function goes here
%   Detailed explanation goes here
fitness_cases = [0 0 1; 0 1 0; 1 0 0; 1 1 1];
out = 0;
for f_case = 1:4
    individual = individual_org;
    index = individual.depthfirstiterator;
    % Leafs
    for i = index
        if individual.isleaf(i)
            if strcmp(individual.get(i),'A')
                individual = individual.set(i,fitness_cases(f_case,1));
            else
                individual = individual.set(i,fitness_cases(f_case,2));
            end
        end
    end
    % Nodes
    flag_finished = 0;
    while ~flag_finished
        flag_finished = 1;
        for i = index
            if strcmp(individual.get(i),'NOT')
                ii = individual.getchildren(i);
                if isnumeric(individual.get(ii))
                    not = individual.get(ii);
                    if not
                        individual = individual.set(i,0);
                    else
                        individual = individual.set(i,1);
                    end
                else
                    flag_finished = 0;
                end
            elseif strcmp(individual.get(i),'AND')
                ii = individual.getchildren(i);
                if isnumeric(individual.get(ii(1)))& isnumeric(individual.get(ii(2)))
                    and = individual.get(ii(1))& individual.get(ii(2));
                    %                 individual = individual.set(i,and);
                    if and
                        individual = individual.set(i,1);
                    else
                        individual = individual.set(i,0);
                    end
                else
                    flag_finished = 0;
                end
            elseif strcmp(individual.get(i),'OR')
                ii = individual.getchildren(i);
                if isnumeric(individual.get(ii(1))) & isnumeric(individual.get(ii(2)))
                    or = individual.get(ii(1))|| individual.get(ii(2));
                    %                 individual = individual.set(i,or);
                    if or
                        individual = individual.set(i,1);
                    else
                        individual = individual.set(i,0);
                    end
                else
                    flag_finished = 0;
                end
            end
        end
    end
    if individual.get(index(1))~=fitness_cases(f_case,3)
        out = out + 1;
    end
end
out = 100*out + depth(individual);
end

