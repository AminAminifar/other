function theta = choose_action( pi,x,phi)
%CHOOSE_ACTION to decide what action should be taken.
epsilon = .9;

[x_i,phi_i,~] = Q_index(x,phi,1);
if rand(1) < epsilon
    action = pi(x_i,phi_i);
else
    action = randi(180,1,1);
end


theta = action;
end

