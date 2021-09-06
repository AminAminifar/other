clc;
close all;
clear all;
global Q;
% Q = zeros(389,180,180);
Q = zeros(777,180,180);
G = zeros(777,180,180);
R = zeros(777,180,180,2);
pi_p = randi(180,777,180);

gamma = .9;

for episode = 1:10
    %initialize s
    a = 0;
    b = 20;
    x = floor((b-a)*rand(1) + a);
    first_x = x;
    y = 0;
    
    a = 0;
    b = 180;
    phi = floor((b-a)*rand(1) + a);
    first_phi = phi;
    tic
    num = 1;
    while ~((x>=10-.1)&&(x<=10+.1)&&...
            (phi>=90-1)&&(phi<=90+1))
        %until s is terminal
        theta = choose_action(pi_p,x,phi);
        %Choose a from s using policy derived from Q
        %         theta = action;
        [ nx,ny,nphi ] = new_state( x,y,phi,theta );
        r = reward(x,phi);
        %take action a observe r and new state
        [x_i,phi_i,action_i] = Q_index(x,phi,theta);
        sequence(num,:) = [x_i,phi_i,action_i];
        %G ? Retun following the first occurence of s,a
        for i=1:num%length(sequence(:,1))
            G(sequence(i,1),sequence(i,2),sequence(i,3)) =...
                G(sequence(i,1),sequence(i,2),sequence(i,3)) +...
                r*(gamma^(num-i));
        end
        %     update_Q(Q,x,phi,theta,r,nx,nphi);
        x = nx;
        y = ny;
        phi = nphi;
        num = num + 1;
    end
    %Append G to Returns(s,a)
    %Q(s,a) ? average(Returns(s,a))
    for i=1:num-1
        var1 = R(sequence(i,1),sequence(i,2),sequence(i,3),1);
        var2 = R(sequence(i,1),sequence(i,2),sequence(i,3),2);
        var3 = G(sequence(i,1),sequence(i,2),sequence(i,3));
        var1 = (var1*var2 + var3)/(var2+1);
        R(sequence(i,1),sequence(i,2),sequence(i,3),2) = var2 + 1;
    end
    Q = R(:,:,:,1);
    %pi_p(s) ? argmax_a Q(s,a)
    for i=1:777
        for j=180
            [~,pi_p(i,j)] = max(Q(i,j,:)) ;
        end
    end
    
    disp(['Episode: ',num2str(episode),char(10),...
        'Start Point:',char(10),...
        'X: ',num2str(first_x),char(9),'phi: ',num2str(first_phi),char(10),...
        'Counted moves: ',num2str(num),char(10),...
        'Time elapsed: ',num2str(toc),char(10)]) ;
    %     num
    %     toc
    %     h = msgbox('Operation Completed');
end
display('The end of training!')
save('pi.mat','pi_p')
display('The variable pi_p saved.')
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
% b = 10;
% x = floor((b-a)*rand(1) + a)
% y = 0;
% a = 0;
% b = 180;
% phi = floor((b-a)*rand(1) + a)
% phi2 = phi;
% if phi2>90
%     phi2 = 180 - mode(phi,90);
% end
% x2 = x + 4*cosd(phi2);
% y2 = y + 4*sind(phi2);
%
% num = 1;
% while ~((x>=10-.1)&&(x<=10+.1)&&...
%         (phi>=90-1)&&(phi<=90+1))
%     %until s is terminal
%     theta = choose_action(pi_p,x,phi);
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
%     pos2(1,num) = x;
%     pos2(2,num) = x2;
%     pos2(3,num) = y;
%     pos2(4,num) = y2;
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
%
%     num = num + 1;
% end
% num
% figure('Name','Last 100 moves')
% plot([pos(1,:) pos(2,:)], [pos(3,:) pos(4,:)])
% hold on
% % plot([pos2(1,:) pos2(2,:)], [pos2(3,:) pos2(4,:)])
% plot([pos2(1,num-1) pos2(2,num-1)], [pos2(3,num-1) pos2(4,num-1)],'r')
%
% % figure2('Name','Last 10 moves')
% % img = imread('truck3.jpg');
% % for i=1:10
% % image([pos2(1,num-11+i) pos2(2,num-11+i)],...
% %     [pos2(3,num-11+i) pos2(4,num-11+i)],img);
% % pause(1);
% % end
