clc;
clear all;
close all;
%% Initial Population
function_set = {'AND','OR','NOT'};
terminal_set = {'A','B'};
general_set = union(function_set,terminal_set);

max_depth = 10;
popsize = 200;
for i = 1:popsize
    pop(i).t = tree(function_set(randi(3)));
end

for  i = 1:popsize
    while pop(i).t.depth<max_depth
        if rand(1)<.8
            index = pop(i).t.depthfirstiterator;
            for j = index
                if pop(i).t.isleaf(j)
                    flag = false;
                    hight = 1;
                    p = j;
                    while pop(i).t.getparent(p)~=0
                        p = pop(i).t.getparent(p);
                        hight = hight + 1;
                    end
                    if hight>=max_depth
                        flag = true;
                    end
                    if strcmp(pop(i).t.get(j),'NOT')
                        if ~flag
                            pop(i).t = pop(i).t.addnode(j,general_set(randi(5)));
                        else
                            pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
                        end
                    elseif strcmp(pop(i).t.get(j),'OR')||strcmp(pop(i).t.get(j),'AND')
                        if ~flag
                            pop(i).t = pop(i).t.addnode(j,general_set(randi(5)));
                            pop(i).t = pop(i).t.addnode(j,general_set(randi(5)));
                        else
                            pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
                            pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
                        end
                    end
                end
            end
        else
            break;
        end
    end
    index = pop(i).t.depthfirstiterator;
    for j = index
        if pop(i).t.isleaf(j)
            if strcmp(pop(i).t.get(j),'NOT')
                pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
            elseif strcmp(pop(i).t.get(j),'OR')||strcmp(pop(i).t.get(j),'AND')
                
                pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
                pop(i).t = pop(i).t.addnode(j,terminal_set(randi(2)));
                
            end
        end
    end
    
    
end
%%

% for i = 1:popsize
%     display('TREE',num2str(i))
%     disp(pop(i).t.tostring)
% end
%% Evaluate initial population
for i = 1:popsize
    pop(i).eval = evaluate(pop(i).t);
end

%% EA Alg.

max_iteration = 30;
Mutation_rate = .4;
CrossOver_rate = .5;
rnd_selection_rate = .1;

temp_pop = pop;
[~,index_pop]=sort([temp_pop.eval]);
pop = temp_pop(index_pop(1:popsize));
best_indv = zeros(popsize,1);
for iteration = 1:max_iteration
    for i = 1:floor(Mutation_rate*popsize)
        %         selected_indvs = sort(randperm(popsize,max(1,floor(rnd_selection_rate*popsize))));
        selected_indvs = randperm(popsize,max(1,floor(rnd_selection_rate*popsize)));
        mutated_pop(i).t = mutate(pop(selected_indvs(1)).t);
        mutated_pop(i).eval = evaluate(mutated_pop(i).t);
    end
    for i = 1:floor(CrossOver_rate*popsize)
        %         selected_indvs = sort(randperm(popsize,max(2,floor(rnd_selection_rate*popsize))));
        selected_indvs = randperm(popsize,max(2,floor(rnd_selection_rate*popsize)));
        [cross_pop(i).t cross_pop(floor(CrossOver_rate*popsize)+i).t] = ...
            crossover(pop(selected_indvs(1)).t,pop(selected_indvs(2)).t);
        cross_pop(i).eval = evaluate(cross_pop(i).t);
        cross_pop(floor(CrossOver_rate*popsize)+i).eval = evaluate(cross_pop(i).t);
    end
    
    temp_pop = [pop, mutated_pop, cross_pop];
    [~,index_pop]=sort([temp_pop.eval]);
    pop = temp_pop(index_pop(1:popsize));
    
    best_indv(iteration) = pop(1).eval;
    figure(1);
    plot(best_indv(1:iteration));
    pause(.05)
end
%% diplay best tree

    display('BEST TREE')
    disp(pop(1).t.tostring)


