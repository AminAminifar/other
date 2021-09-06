clc;
close all;
% clear all;
% global Q;

Q = zeros(5,5,4,2,4);%x y g d a(action)
% Q = ones(5,5,4,2,4)*20;%x y g d a(action)

cum_num = 0;
cum_num_l = 0;
for episode = 1:1000
    %initialize s
    x = randi(5,1);x1 = x;
    y = randi(5,1);y1 = y;
    g = randi(4,1);g1 = g;
    d = 1;
    % h = randi(3,1);
    tic
    num = 0;
    rw = 0;
    while ~((g==1 && x==1 && y==5) ||...
            (g==2 && x==5 && y==5) ||...
            (g==3 && x==1 && y==1) ||...
            (g==4 && x==5 && y==1))
        
        h = randi(3,1);
        %until s is terminal
        action = choose_action(Q,x,y,g,d);
        
        if rand(1)<.4
            if action==1
                action = 3;
            end
        end
        %Choose a from s using policy derived from Q
        [ nx,ny,ng,nd ] = new_state( x,y,g,d,action,h );
        r = reward(nx,ny,ng,d,h,action);
        rw = rw + reward(nx,ny,ng,d,h,action);
        %take action a observe r and new state
        Q(x,y,g,d,action) =...
            update_Q(Q,x,y,g,d,action,r,nx,ny,ng,nd);
        x = nx;
        y = ny;
        d = nd;
        num = num + 1;
    end
    episode_r(episode) = rw;
    disp(['Episode: ',num2str(episode),char(10),...
        'Start Point:',char(10),...
        'X: ',num2str(x1),char(9),'Y: ',num2str(y1),char(10),...
        'G: ',num2str(g-1),char(9),'D: ',num2str(0),char(10),...
        'Counted moves: ',num2str(num),char(10),...
        'Time elapsed: ',num2str(toc),char(10)]) ;
    cum_num = cum_num +num;
    if episode>900
        cum_num_l = cum_num_l +num;
    end
end
disp(['Mean number of moves:',num2str(cum_num/1000)])
disp(['Mean number of moves(Last 100 episodes):',num2str(cum_num_l/100)])
plot(1:1000,episode_r)
title('EPISODE-CUMULATIVE REWARD')
xlabel('EPISODES')
ylabel('CUMULATIVE REWARD')

display('The end of training!')
save('Q.mat','Q')
display('The variable Q saved.')
%%
amp=1;
fs=2000;  % sampling frequency
duration=2;
freq=100;
values=0:1/fs:duration;
sig=amp*sin(2*pi* freq*values);
sound(sig)

%%
display('Starting the GUI:')
GUI()

