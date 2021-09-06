function GUI()
% clc;
% clear all;
% close all;
s = load('pi.mat');
c = struct2cell(s);
pi = cell2mat(c);
figure
% subplot(1,2,1)
mytxt = uicontrol('Style','pushbutton','String','Compute');%,'ButtonDownFcn',@fnc1);
mytxt.Enable = 'Inactive';
mytxt.ButtonDownFcn = @fnc1;
uicontrol('Style','text', 'Position',[10 330 60 20],'String','INPUTS:');
hEdit1 = uicontrol('Style','edit', 'Position',[150 300 60 20], 'String','0');
uicontrol('Style','text', 'Position',[10 300 60 20],'String','X:');
hEdit2 = uicontrol('Style','edit', 'Position',[150 270 60 20], 'String','0');
uicontrol('Style','text', 'Position',[10 270 60 20],'String','PHI:');
uicontrol('Style','text', 'Position',[10 150 60 20],'String','OUTPUTS:');
hEdit3 = uicontrol('Style','edit', 'Position',[150  120 60 20], 'String','');
uicontrol('Style','text', 'Position',[10 120 100 20],'String','Time Elapsed:');
hEdit4 = uicontrol('Style','edit', 'Position',[150  90 60 20], 'String','');
uicontrol('Style','text', 'Position',[10 90 100 20],'String','Number of actions:');
    function fnc1(src,ev)
        %fnc1 Summary of this function goes here
        %   Detailed explanation goes here
        
        
        x = str2double(get(hEdit1, 'String'));
        x = floor(x);
        phi = str2double(get(hEdit2, 'String'));
        phi = floor(phi);
        y = 0;
        x2 = x + 4*cosd(phi);
        y2 = y + 4*sind(phi);
        subplot(1,2,2)
        img = imread('truck4.jpg');
        theta = phi;
        I = img;
        Irot = imrotate(I,theta);
        Mrot = ~imrotate(true(size(I)),theta);
        Irot(Mrot&~imclearborder(Mrot)) = 255;
        img2 = Irot;
        l1 = 2;
        l2 = 4;
        r1 = (l1/2)*((y2-y)/l2);
        r2 = (l1/2)*((x2-x)/l2);
        
        image([x-r1 x2+r1],[y-r2 y2+r2],img2);
        axis([0 20 -5 5])
        %         set(gca,'ydir','normal');
        grid on
        title('Truck in the first place')
        
        
        
        phi2 = phi;
        if phi2>90
            phi2 = 180 - mode(phi,90);
        end
        x2 = x + 4*cosd(phi2);
        y2 = y + 4*sind(phi2);
        
        num = 1;
        tic
        while ~((x>=10-.1)&&(x<=10+.1)&&...
                (phi>=90-1)&&(phi<=90+1))
            %until s is terminal
            theta = choose_action(pi,x,phi);
            %Choose a from s using policy derived from Q
            %         theta = action;
            [ nx,ny,nphi ] = new_state( x,y,phi,theta );
            %     r = reward(x,phi);
            %take action a observe r and new state
            [x_i,phi_i,action_i] = Q_index(x,phi,theta);
            %     Q(x_i,phi_i,action_i) =...
            %         update_Q(Q,x,phi,theta,r,nx,nphi);
            x = nx;
            y = ny;
            phi = nphi;
            phi2 = phi;
            if phi2>90
                phi2 = 180 - mode(phi,90);
            end
            x2 = x + 4*cosd(phi2);
            y2 = y + 4*sind(phi2);
            pos2(1,num) = x;
            pos2(2,num) = x2;
            pos2(3,num) = y;
            pos2(4,num) = y2;
            if num<101
                pos(1,num) = x;
                pos(2,num) = x2;
                pos(3,num) = y;
                pos(4,num) = y2;
            else
                pos(1,:) = circshift(pos(1,:),[0,-1]);
                pos(1,100) = x;
                pos(2,:) = circshift(pos(2,:),[0,-1]);
                pos(2,100) = x2;
                pos(3,:) = circshift(pos(3,:),[0,-1]);
                pos(3,100) = y;
                pos(4,:) = circshift(pos(4,:),[0,-1]);
                pos(4,100) = y2;
            end
            
            num = num + 1;
        end
        t = toc;
        %         num
        figure('Name','Last moves')
        subplot(1,2,1)
        
        plot([pos(1,:) pos(2,:)], [pos(3,:) pos(4,:)])
        title('Trajectory: Last 100 move')
        hold on
        % plot([pos2(1,:) pos2(2,:)], [pos2(3,:) pos2(4,:)])
        plot([pos2(1,num-1) pos2(2,num-1)], [pos2(3,num-1) pos2(4,num-1)],'r')
        hold off
        % figure('Name','Last 100 moves')
        subplot(1,2,2)
        
        axis([-5 25 min(pos(3,:)) max(pos(4,:)) ])
        img = imread('truck4.jpg');
        l1 = 2;
        l2 = 4;
        for i=1:10
            r1 = (l1/2)*((pos2(4,num-11+i)-pos2(3,num-11+i))/l2);
            r2 = (l1/2)*((pos2(1,num-11+i)-pos2(2,num-11+i))/l2);
            z =0;
            if pos2(1,num-11+i)>pos2(2,num-11+i)
                z=90;
            end
            %     img2 = imrotate(img,...
            %         z+asind(((pos2(4,num-11+i)-pos2(3,num-11+i))/l2))...
            %     , 'loose','crop');
            theta = z+asind(((pos2(4,num-11+i)-pos2(3,num-11+i))/l2));
            I = img;
            Irot = imrotate(I,theta);
            Mrot = ~imrotate(true(size(I)),theta);
            Irot(Mrot&~imclearborder(Mrot)) = 255;
            %View 'er
            % imtool(Irot)
            img2 = Irot;
            image([pos2(1,num-11+i)-r1 pos2(2,num-11+i)+r1],...
                [pos2(3,num-11+i)-r2 pos2(4,num-11+i)]+r2,img2);
            axis([min(pos(1,:)) max(pos(2,:))  min(pos(3,:)) max(pos(4,:)) ])
            title('Visual results: Last 10 moves')
            pause(.1);
        end
        set(hEdit3, 'String',t)
        set(hEdit4, 'String',num)
        
        figure('Name','Time Position')
        subplot(1,2,1)
        plot(pos2(1,:),1:num-1)
        title('All moves')
        xlabel('X')
        ylabel('Time')
        subplot(1,2,2)
        plot(pos2(1,num-100:num-1),num-100:num-1)
        title('Last 100 moves')
        xlabel('X')
        ylabel('Time')
        
    end
end
