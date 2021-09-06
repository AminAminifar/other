function GUI()
% clc;
% clear all;
% close all;
s = load('Q.mat');
c = struct2cell(s);
Q = cell2mat(c);
figure
% subplot(1,2,1)
mytxt = uicontrol('Style','pushbutton','String','Compute');%,'ButtonDownFcn',@fnc1);
mytxt.Enable = 'Inactive';
mytxt.ButtonDownFcn = @fnc1;
uicontrol('Style','text', 'Position',[10 330 60 20],'String','INPUTS:');
hEdit1 = uicontrol('Style','edit', 'Position',[150 300 60 20], 'String','1');
uicontrol('Style','text', 'Position',[10 300 60 20],'String','X:');
hEdit2 = uicontrol('Style','edit', 'Position',[150 270 60 20], 'String','1');
uicontrol('Style','text', 'Position',[10 270 60 20],'String','Y:');
hEdit5 = uicontrol('Style','edit', 'Position',[150 240 60 20], 'String','1');
uicontrol('Style','text', 'Position',[10 240 60 20],'String','G:');
uicontrol('Style','text', 'Position',[10 150 60 20],'String','OUTPUTS:');
hEdit3 = uicontrol('Style','edit', 'Position',[150  120 60 20], 'String','');
uicontrol('Style','text', 'Position',[10 120 100 20],'String','Time Elapsed:');
hEdit4 = uicontrol('Style','edit', 'Position',[150  90 60 20], 'String','');
uicontrol('Style','text', 'Position',[10 90 100 20],'String','Number of actions:');
    function fnc1(src,ev)
        %fnc1 Summary of this function goes here
        %   Detailed explanation goes here
        
        %         close all hidden
        
        
        x = str2double(get(hEdit1, 'String'));
        x = floor(x);
        y = str2double(get(hEdit2, 'String'));
        y = floor(y);
        g = str2double(get(hEdit5, 'String'));
        g = floor(g);
        d = 1;
        
        if g == 1%G0
            xg= 1 ; yg = 5;
        elseif g == 2%G1
            xg = 5;  yg = 5;
        elseif g == 3%G2
            xg = 1;  yg = 1;
        elseif g == 4%G3
            xg = 5; yg = 1;
        end
        
        x2 = x - 1;
        y2 = y - 1;
        subplot(1,2,2)
        img = imread('WALL.E3.jpg');
        
        
        
        image([x2 x],[y2 y],img);
        %         Set (gca,'Ydir','reverse')
        axis([0 5 0 5])
        %           set(gca,'YDir','normal')
        axis xy;
        %         hold on
        %   set(gcf,'Color',[1 1 1])
        gx = [xg-1 xg xg xg-1];
        gy = [yg-1 yg-1 yg yg];
        patch(gx,gy, 'green','facealpha', .2,'edgecolor', 'none');
        rx = [1 2 2 1];
        ry = [4 4 5 5];
        patch(rx,ry, 'blue','facealpha', .2,'edgecolor', 'none');
        %         rectangle('Position',[xg-1 yg-1 .1 .1],'FaceColor',[0 .7 .4],'FaceAlpha',.2)
        rectangle('Position',[.9 3 .2 2],'FaceColor',[1 0 0])
        rectangle('Position',[1.9 4 .2 1],'FaceColor',[1 0 0])
        %                     hold on
        grid on
        title('First place')
        %%
        num = 1;
        tic;
        fig2 = figure;
        gx = [xg-1 xg xg xg-1];
        gy = [yg-1 yg-1 yg yg];
        patch(gx,gy, 'green','facealpha', .2,'edgecolor', 'none');
        rx = [1 2 2 1];
        ry = [4 4 5 5];
        patch(rx,ry, 'blue','facealpha', .2,'edgecolor', 'none');
        %         rectangle('Position',[xg-1 yg-1 .1 .1],'FaceColor',[0 .7 .4],'FaceAlpha',.2)
        rectangle('Position',[.9 3 .2 2],'FaceColor',[1 0 0])
        rectangle('Position',[1.9 4 .2 1],'FaceColor',[1 0 0])
        hold on
        grid on
        rew(1) = 10;
        rew2(1) = 10;
        
        while ~((g==1 && x==1 && y==5) ||...
                (g==2 && x==5 && y==5) ||...
                (g==3 && x==1 && y==1) ||...
                (g==4 && x==5 && y==1))
            
            h = randi(3);
            if h == 1%H1
                xh = 3 ; yh = 4;
            elseif h == 2%H2
                xh = 2;  yh = 2;
            elseif h == 3%H3
                xh = 4;  yh = 2;
            end
            pos(num).hole = [xh,yh];
            %             hx = [xh-1 xh xh xh-1];
            %             hy = [yh-1 yh-1 yh yh];
            %            h1 =  patch(hx,hy, 'yellow','facealpha', .2,'edgecolor', 'none');
            
            %until s is terminal
            action = choose_action(Q,x,y,g,d);
            
            if rand(1)<.4
                if action==1
                    action = 3;
                end
            end
            %Choose a from s using policy derived from Q
            [ nx,ny,ng,nd ] = new_state( x,y,g,d,action,h );
            if num == 1
                rew(num) = reward(nx,ny,ng,d,h,action);
            else
                rew(num) =rew(num-1)+ reward(nx,ny,ng,d,h,action);
            end
            rew2(num) = reward(nx,ny,ng,d,h,action);
            %             if reward(nx,ny,ng,nd,h,action)==10
            %                 display('yey')
            %             end
            %take action a observe r and new state
            %             Q(x,y,g,d,action) =...
            %                 update_Q(Q,x,y,g,d,action,r,nx,ny,ng,nd);
            journey(num).x = x;
            journey(num).y = y;
            journey(num).Q = Q(x,y,g,d,action);
            x = nx;
            y = ny;
            d = nd;
            
            pos(num).state = [x,y,d];

            num = num + 1;
            
            %             x2 = x - 1;
            %             y2 = y - 1;
            %      set(h1,'Visible','off')
            %           h2 =  image([x2 x],[y2 y],img);
            % % hold off
            %                    axis([0 5 0 5])
            %                    axis xy;
            %             pause(.01)
            %                     grid on
            %                       set(h2,'Visible','off')
            %                       set(h1,'Visible','off')
        end
        %Last move
        journey(num).x = x;
        journey(num).y = y;
        journey(num).Q = Q(x,y,g,d,action);
        
        num = num - 1;
        
        t = toc;
        set(hEdit3, 'String',t)
        set(hEdit4, 'String',num)
        %%
        if num>0
            if num>100
                it = 100;
            else
                it = num;
            end
            for i=1:it
                
                hx = [pos(num-it+i).hole(1)-1 pos(num-it+i).hole(1)...
                    pos(num-it+i).hole(1) pos(num-it+i).hole(1)-1];
                hy = [pos(num-it+i).hole(2)-1 pos(num-it+i).hole(2)-1 ...
                    pos(num-it+i).hole(2) pos(num-it+i).hole(2)];
                h1 =  patch(hx,hy, 'yellow','facealpha', .2,'edgecolor', 'none');
                h2 =  image([pos(num-it+i).state(1)-1 pos(num-it+i).state(1)],...
                    [pos(num-it+i).state(2)-1 pos(num-it+i).state(2)],img);
                if pos(num-it+i).state(3)==2
%                                     display('Damaged');
                    dx = [pos(num-it+i).state(1)-1 pos(num-it+i).state(1)...
                        pos(num-it+i).state(1) pos(num-it+i).state(1)-1];
                    dy = [pos(num-it+i).state(2)-1 pos(num-it+i).state(2)-1 ...
                        pos(num-it+i).state(2) pos(num-it+i).state(2)];
                    h3 = patch(dx,dy, 'red','facealpha', .4,'edgecolor', 'none');
                    %             else
                    %                 display('In good condition')
                end
                
                if i>1 && (pos(num-it+i-1).state(3)~=pos(num-it+i).state(3))
                    if pos(num-it+i).state(3)==2;
                        display('Damaged');
                        [yData,FsData] = audioread('thumb2.mp3');
                        sound(yData,FsData)
                        pause(1)
                    else
                        display('In good condition')
                        [yData,FsData] = audioread('thumb.mp3');
                        sound(yData,FsData)
                        pause(2.405)
                    end
                end
                
                % hold off
                axis([0 5 0 5])
                axis xy;
                pause(.1)
                grid on
                if i~=it
                    set(h2,'Visible','off')
                    set(h1,'Visible','off')
                    if pos(num-it+i).state(3)==2
                        set(h3,'Visible','off')
                    end
                else
                    gx = [xg-1 xg xg xg-1];
                    gy = [yg-1 yg-1 yg yg];
                    patch(gx,gy, 'green','facealpha', .2,'edgecolor', 'none');
                    
                    display('REACHED GOAL!')
                    [yData,FsData] = audioread('smb3_enter_level.wav');
                    sound(yData,FsData)
                    pause(1)
                end
            end
            
            figure;
            subplot(1,2,1)
            plot(1:num,rew(1:num))
            title('Cumulative sum of rewards in journey')
            xlabel('STEPS')
            ylabel('CUMULATIVE REWARD')
            subplot(1,2,2)
            
            plot(1:num,rew2(:));
            title('Rewards in each step')
            
            xlabel('STEPS')
            ylabel('REWARD')
            
            %%
            %             table(journey(:).x , journey(:).y , journey(:).Q  )
            f = figure();
            
            % create the data
            d = [journey(:).x ;journey(:).y ;journey(:).Q ]';
            
            % Create the column and row names in cell arrays
            cnames = {'X','Y','Q(s,a)'};
            rnames = {1:num+1};
            
            % Create the uitable
            t = uitable(f,'Data',d,...
                'ColumnName',cnames,...
                'RowName',rnames);
            
            % Set width and height
            t.Position(3) = t.Extent(3);
            t.Position(4) = t.Extent(4);
        else
            close(fig2)
        end
        
        %%
        
    end
end
