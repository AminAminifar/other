clc;
clear;
close all;

% Question 2

%% Create Data
numberOfData = 1000;
x = -1 + 2*rand([2,numberOfData])';

plot(x(:,1),x(:,2),'.')
hold on
%% Center Initialization
numberOfCenter = 4;
center = zeros(4,2);
Colors=hsv(numberOfCenter);

for c=1:numberOfCenter
    center(c,:) = x(c,:);
    plot(center(c,1),center(c,2),'ks','MarkerSize',12,'MarkerFaceColor',Colors(c,:))
end
legend('Data','Center A','Center B','Center C','Center D');
%% K-Means Algorithm
IDX = zeros(1000,1);
epsilon = .01;
eps_vec = ones(1,numberOfCenter);
eps = 1;
iter = 0;
while eps >= epsilon
    iter = iter + 1;
    
    % Assignment step
    for i=1: numberOfData
        d1 = norm(center(1,:) - x(i,:),2);
        d2 = norm(center(2,:) - x(i,:),2);
        d3 = norm(center(3,:) - x(i,:),2);
        d4 = norm(center(4,:) - x(i,:),2);
        distance = [d1; d2; d3; d4];
        [dist index] = min(distance);
        IDX(i) = index;
    end
    hold off
    plot(x(IDX==1,1),x(IDX==1,2),'.','Color',Colors(1,:))
    hold on
    plot(x(IDX==2,1),x(IDX==2,2),'.','Color',Colors(2,:))
    plot(x(IDX==3,1),x(IDX==3,2),'.','Color',Colors(3,:))
    plot(x(IDX==4,1),x(IDX==4,2),'.','Color',Colors(4,:))
    % Update step
    for i=1:numberOfCenter
        IDXCI = (IDX == i);
        nIDXCI = sum(IDXCI);
        datacluster_i = x(IDXCI,:);
        xi_sum = sum(datacluster_i(:,1));
        xj_sum = sum(datacluster_i(:,2));
        new_center_i = [xi_sum xj_sum]/nIDXCI;
        eps_vec(i) = norm(center(i,:) - new_center_i,2);
        center(i,:) = new_center_i;
    end
    % Update Epsilon
    eps = max(eps_vec);
    disp(['[X] Iteration : ', num2str(iter), ' ,Error : ', num2str(eps)]);
    for c=1:numberOfCenter
        plot(center(c,1),center(c,2),'ks','MarkerSize',12,'MarkerFaceColor',Colors(c,:))
    end
    pause(1);
end