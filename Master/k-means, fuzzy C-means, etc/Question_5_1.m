clc;
clear;
close all;

% Question 5-1

%% Create Data
x1 = load('Data/petal_length.mat');
x1 = x1.petal_length;
x2 = load('Data/petal_width.mat');
x2 = x2.petal_width;
x3 = load('Data/sepal_length.mat');
x3 = x3.sepal_length;
x4 = load('Data/sepal_width.mat');
x4 = x4.sepal_width;

Type1 = [x1(:,1) x2(:,1) x3(:,1) x4(:,1)];
Type2 = [x1(:,2) x2(:,2) x3(:,2) x4(:,2)];
Type3 = [x1(:,3) x2(:,3) x3(:,3) x4(:,3)];

x = [Type1;Type2;Type3];
numberOfData = size(x,1);

%% Center Initialization
numberOfCenter = 3;
center = zeros(3,4);

c1_index = randi(numberOfData);
c2_index = randi(numberOfData);
c3_index = randi(numberOfData);
center(1,:) = x(c1_index,:);
center(2,:) = x(c2_index,:);
center(3,:) = x(c3_index,:);

%% K-Means Algorithm
U = zeros(numberOfData,1);
epsilon = 10^-4;
eps_vec = ones(1,numberOfCenter);
eps = 1;
iter =0;
while eps >= epsilon
    iter = iter + 1;
    % Assignment step
    for i=1: numberOfData
        d1 = norm(center(1,:) - x(i,:),2);
        d2 = norm(center(2,:) - x(i,:),2);
        d3 = norm(center(3,:) - x(i,:),2);
        distance = [d1; d2; d3];
        [dist index] = min(distance);
        U(i) = index;
    end
    % Update step
    for i=1:numberOfCenter
        IDXCI = (U == i);
        nIDXCI = sum(IDXCI);
        datacluster_i = x(IDXCI,:);
        x_sum = sum(datacluster_i);
        new_center_i = x_sum./nIDXCI;
        eps_vec(i) = norm(center(i,:) - new_center_i,2);
        center(i,:) = new_center_i;
    end
    % Update Epsilon
    eps = max(eps_vec);
    disp(['[X] Iteration : ', num2str(iter), ' ,Error : ', num2str(eps)]);
end