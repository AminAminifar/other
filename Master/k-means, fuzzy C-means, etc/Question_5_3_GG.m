clc;
clear;
close all;

% Question 5-4(GG)

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
n = size(x,2);
%% Center Initialization
numberOfCenter = 3;
center = zeros(3,4);

c1_index = randi(numberOfData);
c2_index = randi(numberOfData);
c3_index = randi(numberOfData);
center(1,:) = x(c1_index,:);
center(2,:) = x(c2_index,:);
center(3,:) = x(c3_index,:);

%% Membership Function Initialization
m = 2;
w = 2;
U = rand(numberOfData,numberOfCenter);
gamma = .4;
betta = .3;
ro = .3;
ep = .00001;
% normalization
for i=1: numberOfData
    U(i,:) = U(i,:)./sum(U(i,:));
end

%% GK Algorithm
epsilon = 10^-4;
eps_vec = ones(1,numberOfCenter);
eps = 1;
F = zeros(4,1);
iter = 0;
while eps >= epsilon
    iter = iter + 1;
    % 1- Update centers
    for c=1:numberOfCenter
        xy_sum = 0;
        um_sum = 0;
        for n=1: numberOfData
            xy_sum = xy_sum + (U(n,c)^m) * x(n,:);
            um_sum = um_sum + U(n,c)^m;
        end
        new_center_i = xy_sum/um_sum;
        eps_vec(c) = norm(center(c,:) - new_center_i,2);
        center(c,:) = new_center_i;
    end
    
    % 2- Calculate distance matrix
    for c=1:numberOfCenter
        ud_sum = 0;
        um_sum = 0;
        for n=1: numberOfData
            ud_sum = ud_sum + (U(n,c)^w) * (x(n,:) - center(c,:))*(x(n,:) - center(c,:))';
            um_sum = um_sum + U(n,c)^w;
        end
        new_f_i = ud_sum/um_sum;
        F(c,:) = new_f_i;
    end
    % Update Membership function
    for i=1: numberOfData
        mu1i = (1/numberOfData)*sum(U(:,1));
        d1i = ((2*pi)^(n/2)*sqrt(det(F(1))))*exp((x(i,:) - center(1,:)) * F(1)^-1 *(x(i,:) - center(1,:))')+ep;
        mu2i = (1/numberOfData)*sum(U(:,2));
        d2i = ((2*pi)^(n/2)*sqrt(det(F(1))))*exp((x(i,:) - center(2,:)) * F(2)^-1 *(x(i,:) - center(2,:))')+ep;
        mu3i = (1/numberOfData)*sum(U(:,2));
        d3i = ((2*pi)^(n/2)*sqrt(det(F(1))))*exp((x(i,:) - center(3,:)) * F(3)^-1 *(x(i,:) - center(3,:))')+ep;
        
        U(i,1) = 1/((d1i/d1i)^(2/(m - 1)) + (d1i/d2i)^(2/(m - 1))+(d1i/d3i)^(2/(m - 1)));
                    
        U(i,2) = 1/((d2i/d1i)^(2/(m - 1)) + (d2i/d2i)^(2/(m - 1))+(d2i/d3i)^(2/(m - 1)));
                    
        U(i,3) = 1/((d3i/d1i)^(2/(m - 1)) + (d3i/d2i)^(2/(m - 1))+(d3i/d3i)^(2/(m - 1)));
    end
    for i=1: numberOfData
        U(i,:) = U(i,:)./sum(U(i,:));
    end
    % Update Epsilon
    eps = max(eps_vec);
    disp(['[X] Iteration : ', num2str(iter), ' ,Error : ', num2str(eps)]);
end