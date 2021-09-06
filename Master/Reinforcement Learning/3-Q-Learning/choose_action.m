function direction = choose_action( Q,x,y,g,d )
%CHOOSE_ACTION Summary of this function goes here
%   Direction 1? up 2? right 3? down 4? left
epsilon = .2;%explore rate 
expt_r = 1 - epsilon;


if rand(1) < expt_r
    [~,I] = max(Q(x,y,g,d,:));
    if range(Q(x,y,g,d,:)) == 0
        I = randi(4,1);
    end
    action = I;
else
    I = randi(4,1);
    action = I;
end


direction = action;
end

