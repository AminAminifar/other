clc;
close all;
% clear all;
global Q;
% Q = zeros(389,180,180);
Q = zeros(777,180,180);
% x = 1;
% y = 2;
% phi = 30;
% theta = 80;
%
% next_x(x,phi,theta)
% next_y(y,phi,theta)
% next_phi(phi,theta)


for episode = 1:10
    %initialize s
    a = 0;
    b = 20;
    x = floor((b-a)*rand(1) + a);
    first_x = x;
    y = 0;
    %     a = 0;
    %     b = 20;
    %     y = (b-a)*rand(1) + a;
    %It is not important to give a randome number
    %to y, since we do not consider it.
    %We can give a fixed number to y.
    a = 0;
    b = 180;
    phi = floor((b-a)*rand(1) + a);
    first_phi = phi;
    tic
    num = 0;
    while ~((x>=10-.1)&&(x<=10+.1)&&...
            (phi>=90-1)&&(phi<=90+1))
        %until s is terminal
        theta = choose_action(Q,x,phi);
        %Choose a from s using policy derived from Q
        %         theta = action;
        [ nx,ny,nphi ] = new_state( x,y,phi,theta );
        r = reward(x,phi);
        %take action a observe r and new state
        [x_i,phi_i,action_i] = Q_index(x,phi,theta);
        Q(x_i,phi_i,action_i) =...
            update_Q(Q,x,phi,theta,r,nx,nphi);
        x = nx;
        y = ny;
        phi = nphi;
        num = num + 1;
    end
        disp(['Episode: ',num2str(episode),char(10),...
        'Start Point:',char(10),...
        'X: ',num2str(first_x),char(9),'phi: ',num2str(first_phi),char(10),...
        'Counted moves: ',num2str(num),char(10),...
        'Time elapsed: ',num2str(toc),char(10)]) ;
%     disp(['Episode: ',num2str(episode)]);
%     num
%     toc
    %     h = msgbox('Operation Completed');
end


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
% a = 0;
% b = 20;
% x = floor((b-a)*rand(1) + a)
% y = 0;
% % a = 0;
% % b = 20;
% % y = (b-a)*rand(1) + a;
% %It is not important to give a randome number
% %to y, since we do not consider it.
% %We can give a fixed number to y.
% a = 0;
% b = 180;
% phi = floor((b-a)*rand(1) + a)
% phi2 = phi;
% if phi2>90
%     phi2 = 180 - mode(phi,90);
% end
% x2 = x + 4*cosd(phi2);
% y2 = y + 4*sind(phi2);
% % sqrt((x-x2)^2 + (y-y2)^2)
% % plot([x x2], [y y2])
% % axis([-10 30 -100 100])
% % pause(.05)
% 
% % hold on
% num = 1;
% while ~((x>=10-.1)&&(x<=10+.1)&&...
%         (phi>=90-1)&&(phi<=90+1))
%     %until s is terminal
%     theta = choose_action(Q,x,phi);
%     %Choose a from s using policy derived from Q
%     %         theta = action;
%     [ nx,ny,nphi ] = new_state( x,y,phi,theta );
%     %     r = reward(x,phi);
%     %take action a observe r and new state
%     [x_i,phi_i,action_i] = Q_index(x,phi,theta);
%     %     Q(x_i,phi_i,action_i) =...
%     %         update_Q(Q,x,phi,theta,r,nx,nphi);
%     x = nx;
%     y = ny;
%     phi = nphi;
%     phi2 = phi;
%     if phi2>90
%         phi2 = 180 - mode(phi,90);
%     end
%     x2 = x + 4*cosd(phi2);
%     y2 = y + 4*sind(phi2);
%         pos2(1,num) = x;
%         pos2(2,num) = x2;
%         pos2(3,num) = y;
%         pos2(4,num) = y2;
%     if num<101
%         pos(1,num) = x;
%         pos(2,num) = x2;
%         pos(3,num) = y;
%         pos(4,num) = y2;
%     else
%         pos(1,:) = circshift(pos(1,:),[0,-1]);
%         pos(1,100) = x;
%         pos(2,:) = circshift(pos(2,:),[0,-1]);
%         pos(2,100) = x2;
%         pos(3,:) = circshift(pos(3,:),[0,-1]);
%         pos(3,100) = y;
%         pos(4,:) = circshift(pos(4,:),[0,-1]);
%         pos(4,100) = y2;
%     end
%     %     sqrt((x-x2)^2 + (y-y2)^2)
%     %         plot([x x2], [y y2])
%     %         axis([-10 30 -10 500])
%     %         pause(.0005)
%     num = num + 1;
% end
% num
% figure('Name','Last 100 moves')
% plot([pos(1,:) pos(2,:)], [pos(3,:) pos(4,:)])
% hold on
% % plot([pos2(1,:) pos2(2,:)], [pos2(3,:) pos2(4,:)])
% plot([pos2(1,num-1) pos2(2,num-1)], [pos2(3,num-1) pos2(4,num-1)],'r')
