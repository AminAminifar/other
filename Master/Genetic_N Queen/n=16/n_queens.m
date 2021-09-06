clc
clear all;
close all

pop_size = 200;
performance_num = 1;
n = 16;%8 16 32
max_iteration = 100;
mutation_probability = .4;
recombination_probability = .9;
pop.individual = repmat([], pop_size, 1);
pop.fitness = repmat([], pop_size, 1);

num_mutete = ceil(pop_size*mutation_probability);
num_recombination = ceil(pop_size*recombination_probability);
f1 = figure('name','progress');
for performance=1:performance_num
    
    for i=1:pop_size
    pop(i).individual = randperm(n);
    pop(i).fitness = fitness(pop(i).individual);
    end
    
    termination = false;
    episode = 0;
    while ~(termination||episode==max_iteration)
temp_pop = pop;
        %% recombination
        
        for i=1:num_recombination
            [indv_index1, indv_index2] = selection(temp_pop);

            [ temp_pop(pop_size+2*i-1).individual, temp_pop(pop_size+2*i).individual ] =...
                recombination( temp_pop(indv_index1).individual,...
                temp_pop(indv_index2).individual );
            

            
            temp_pop(pop_size+2*i-1).fitness =...
                fitness(temp_pop(pop_size+2*i-1).individual);
            temp_pop(pop_size+2*i).fitness =...
                fitness(temp_pop(pop_size+2*i).individual);
            
        end
        %% mutate
        
        for i=1:num_mutete
            index = randi(length(temp_pop));
            temp_pop(index).individual =...
                mutation( temp_pop(index).individual );
            
            temp_pop(index).fitness =...
                fitness(temp_pop(index).individual);
        end
        %% generation selection
%         for i=1:length(temp_pop)
%             temp_pop(i).fitness = fitness(temp_pop(i).individual);
%         end

    [~, indeces] = sort([temp_pop.fitness],'descend');
    pop = temp_pop(indeces(1:pop_size));
    clear temp_pop
    episode = episode + 1;
    fitness_val(episode) =  pop(1).fitness;
    plot(1:episode,fitness_val(1:episode),'b');
    hold on
    pause(.005)
    if pop(1).fitness==0
        termination = true;
        board = pop(1).individual;
    end
    end

end
%% 

f2 = figure('name','Board');

grid on
for j=1:n
    for i=1:n
        if mod(j,2) == 0
        if mod(i,2) == 0
            xw = i;
            yw = j;
            wx = [xw-1 xw xw xw-1];
            wy = [yw-1 yw-1 yw yw];
            patch(wx,wy, 'black','facealpha', .2,'edgecolor', 'none');
        end
        else
            if mod(i,2) == 1
            xw = i;
            yw = j;
            wx = [xw-1 xw xw xw-1];
            wy = [yw-1 yw-1 yw yw];
            patch(wx,wy, 'black','facealpha', .2,'edgecolor', 'none');
            end
        end
    end
end
hold on
%  img = imread('queen.jpg');
for i=1:n
    img = imread('queen.jpg');
    x1 = board(i);
    y1 = i;
    x1_2 = x1 - 1;
    y1_2 = y1 - 1;
    im(i) =  image([x1_2 x1],[y1_2 y1],img);
    hold on
end
axis([0 n 0 n])
