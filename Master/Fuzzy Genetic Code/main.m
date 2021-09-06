clc;
close all;
clear;

%% Load Data
A = xlsread('Weather information.xlsx','2008');
B = xlsread('Weather information.xlsx','2009');
C = xlsread('Weather information.xlsx','2010');
X = [A; B; C];
X(:,1:2) = [];

%% Data Normalization 
Xmax = max(X);
Xmin = min(X);
X = (2*(X - Xmin)/(Xmax - Xmin))-1;
%% Prepare Data

disp('Data Pre-Proccessing ...');

nData = size(X,1);
Data = zeros(nData-3, 4);

for i = 1:nData-3
    Data(i,:) = X(i:i+3)';
end

index = randperm(nData-3);
TrainDataSize = round((nData*70)/100);
TrainData = Data(index(1:TrainDataSize),:);
TestData = Data(index(TrainDataSize:nData-3),:);

c = [-1, -0.5, 0, 0.5, 1];
sigma = 0.3;

%% GA Parameters
MaxIt=50;      % Maximum Number of Iterations

nPop=20;        % Population Size

pc=0.8;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.4;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu=0.02;         % Mutation Rate

%% Generate Population
disp('Generate Population ...');
pop.pos=[];
pop.cost=[];

popall=repmat(pop,[nPop 1]);

for i=1:nPop
    chromosome = randi(5,125,4);
    chromosome = reshape(chromosome,1,4*125);
    chromosome = Reformer( chromosome );
    popall(i).pos = chromosome;
    popall(i).cost = CostFunction(TrainData, popall(i).pos, c, sigma);
end

% Sort Population
Costs = [popall.cost];
[Costs, SortOrder] = sort(Costs);
popall = popall(SortOrder);

% Store Best Solution
BestSol = popall(1);

% Array to Hold Best Cost Values
BestCost = zeros(MaxIt,1);

%% Main Loop
disp('Main Loop Algorithm ...');
for it=1:MaxIt
    
    % Crossover
    popc=repmat(pop,nc/2,2);
    for k=1:nc/2
        i1=randi([1 nPop]);
        i2=randi([1 nPop]);
        
        % Select Parents
        p1=popall(i1);
        p2=popall(i2);
        
        % Apply Crossover
        [popc(k,1).pos popc(k,2).pos] = DoublePointCrossover(p1.pos,p2.pos);
        popc(k,1).pos = Reformer( popc(k,1).pos );
        popc(k,2).pos = Reformer( popc(k,2).pos );
        % Evaluate Offsprings
        popc(k,1).cost = CostFunction(TrainData, popc(k,1).pos, c, sigma);
        popc(k,2).cost = CostFunction(TrainData ,popc(k,2).pos, c, sigma);
    end
    popc=popc(:);
    
    % Mutation
    popm = repmat(pop,nm,1);
    for k=1:nm
        % Select Parent
        i=randi([1 nPop]);
        p=popall(i);
        
        % Apply Mutation
        popm(k).pos=Mutate(p.pos,mu);
        popm(k).pos = Reformer(popm(k).pos);
        % Evaluate Mutant
        popm(k).cost = CostFunction(TrainData, popm(k).pos, c, sigma);
    end
    
    % Create Merged Population
    popall=[popall
            popc
            popm];%#ok
     
    % Sort Population
    Costs=[popall.cost];
    [Costs, SortOrder]=sort(Costs);
    popall=popall(SortOrder);
    
    % Truncation
    popall=popall(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=popall(1);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it),', Best Cost = ' num2str(BestCost(it))]);
end

%% ResultsbestsRule = BestSol;
Testsize = size(TestData,1);
Trainsize = size(TrainData,1);

testErr = CostFunction(TrainData, BestSol.pos, c, sigma);
trainErr = CostFunction(TestData, BestSol.pos, c, sigma);
disp(['MSE Test : ' num2str(testErr)]);
disp(['MSE Train : ' num2str(trainErr)]);

test_output = zeros(1,Testsize);
train_output = zeros(1,Trainsize);
for i = 1:Testsize
    test_output(i) = Infererece(TestData(i,:), BestSol.pos, c, sigma);
end

output = zeros(1,Trainsize);
for i = 1:Trainsize
    train_output(i) = Infererece(TrainData(i,:), BestSol.pos, c, sigma);
end

figure
subplot(2,1,1);
plot(1:1:Testsize,TestData(:,4))
title('Predict Test Data')
hold on
plot(1:1:Testsize,test_output,'r--')
legend('Real Data', 'Test Data')
xlabel('x')
ylabel('y')

subplot(2,1,2);
plot(1:1:Trainsize,TrainData(:,4))
hold on
plot(1:1:Trainsize,train_output,'r--')
title('Predict Train Data')
legend('Real Data', 'Train Data')
xlabel('x')
ylabel('y')

BestSol = BestSol.pos;
for i=1:4:numel(BestSol)
    disp(['R' num2str(floor(i/4)),' : ' num2str(BestSol(i:i+3))])
end
