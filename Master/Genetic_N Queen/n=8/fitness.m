function [ fitness_value ] = fitness( individual )
%Computes the fitness of an individual.
fitness_value = 0;
% queen(1:length(individual)) = [];

for i=1:length(individual)
queen(i).pos = [i, individual(i)];
end

for i=1:length(individual)
    for j=setdiff(1:length(individual), i)
%         if queen(j).pos(1)==queen(i).pos(1)
%             fitness_value = fitness_value - 1;
%         end
%         if queen(j).pos(2)==queen(i).pos(2)
%             fitness_value = fitness_value - 1;
%         end
        x_diff = queen(j).pos(1)-queen(i).pos(1);
        y_diff = queen(j).pos(2)-queen(i).pos(2);
        
       if bi2de(acosd(abs(x_diff/y_diff))==[45, 90, 0])>=1
           fitness_value = fitness_value - 1;
       end
        
    end
end

end

