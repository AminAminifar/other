clc;
clear;
close all;

% Question 4-1

%% Create Data
x = [-2 0;1 0;0 0;2 0;2 -1;1.5 .5;1.5 0;1.5 -.5;-2 1;2 1;-2 -1;-1.5 .5;-1.5 0;-1.5 -.5;-1 0];
numberOfData = size(x,1);

plot(x(:,1),x(:,2),'.')
hold on
%% Center Initialization
numberOfCenter = 2;
center = zeros(2,2);
Colors=hsv(numberOfCenter);

for c=1:numberOfCenter
    center(c,:) = x(c,:);
    plot(center(c,1),center(c,2),'ks','MarkerSize',12,'MarkerFaceColor',Colors(c,:))
end
legend('Data','Center A','Center B');
%% K-Means Algorithm
U = zeros(numberOfData,1);
epsilon = 0.01;
eps_vec = ones(1,numberOfCenter);
eps = 1;

while eps >= epsilon
    % Assignment step
    for i=1: numberOfData
        d1 = norm(center(1,:) - x(i,:),2);
        d2 = norm(center(2,:) - x(i,:),2);
        distance = [d1; d2];
        [dist index] = min(distance);
        U(i) = index;
    end
    hold off
    plot(x(U==1,1),x(U==1,2),'.','Color',Colors(1,:))
    hold on
    plot(x(U==2,1),x(U==2,2),'.','Color',Colors(2,:))
    % Update step
    for i=1:numberOfCenter
        IDXCI = (U == i);
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
    for c=1:numberOfCenter
        plot(center(c,1),center(c,2),'ks','MarkerSize',12,'MarkerFaceColor',Colors(c,:))
    end
    pause(2);
end