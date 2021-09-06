function [ranking] = optimal_mRMR(X_train,Y_train)

    
    num_selected_features=1;
    num_bins=10;
 
    
    X_train = (X_train - ones(size(X_train,1),1)*mean(X_train))*diag(1./std(X_train));
    
    
    X_train = round(X_train*num_bins)/num_bins;

    num_features = size(X_train,2);

    c=zeros(1,num_features);
    for i=1:num_features
       c(i) = mutualinformation(X_train(:,i), Y_train);
    end

    a=zeros(num_features,num_features);
    for i=1:num_features
        for j=i:num_features
              a(i,j) = mutualinformation(X_train(:,i), X_train(:,j));
              a(j,i) = a(i,j);
        end
    end

    % code here
    %ranking=1:num_features;
    c = -c';
    a = 2*a;
    opts = optimset('Algorithm','interior-point-convex','Display','off');%'interior-point-convex','active-set'
    Aeq = ones(1,num_features);
    beq = 1;
    lb = zeros(num_features,1);
%     [x fval eflag output lambda] = ...
      X =  quadprog(a,c,[],[],Aeq,beq,lb,[],[],opts);
     [~,ranking] = sort(X,'descend');
%      length (intersect(ranking(1:10),1:10))
end


