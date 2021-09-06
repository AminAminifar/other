function [ chromosome ] = Reformer( chromosome )
    
chromosome = reshape(chromosome,4,125);

    inconsistency = 1;
    while inconsistency ~=0
        inconsistency = 0;
        n = size(chromosome ,1);
        for i = 1:n
           c1 = chromosome(i, :);
           for j = 1:n
               c2 = chromosome(j, :);
               if (c1(1) == c2(1) && c1(2) == c2(2) && c1(3) == c2(3) && c1(4) ~= c2(4))
                   v = randi(5);
                   ix = randi(3);
                   chromosome(i, ix) = v;
                   inconsistency = inconsistency + 1;
               end
           end
        end
    end
    
    chromosome = reshape(chromosome,1,4*125);
end

