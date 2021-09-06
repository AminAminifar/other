clear;

num_samples=1000;
num_features=20;


%training set (example)
X_train=(rand(num_samples,num_features)-0.5)*2;
for i=1:num_samples;
    if(sqrt(X_train(i,1)^2+X_train(i,2)*X_train(i,3)-X_train(i,4)^2+X_train(i,5)^3-X_train(i,6)^3+X_train(i,7)*X_train(i,8)*X_train(i,9)*X_train(i,10))>0.0)
    %if(X_train(i,1)+X_train(i,2)-X_train(i,3)-X_train(i,4)>0)        
        Y_train(i)=1;
    else
        Y_train(i)=-1;
    end
end

%test set (example)
X_test=(rand(1000,num_features)-0.5)*2;
for i=1:1000
    if(sqrt(X_test(i,1)^2+X_test(i,2)*X_test(i,3)-X_test(i,4)^2+X_test(i,5)^3-X_test(i,6)^3+X_test(i,7)*X_test(i,8)*X_test(i,9)*X_test(i,10))>0.0)
    %if(X_train(i,1)+X_train(i,2)-X_train(i,3)-X_train(i,4)>0)        
        Y_test(i)=1;
    else
        Y_test(i)=-1;
    end
end

ratio_train=sum(Y_train>0)/length(Y_train)
ratio_test=sum(Y_test>0)/length(Y_test)

%%
tic
ind=optimal_mRMR(X_train,Y_train')
toc

%%

num=5;

%now only use the important features according to the optimal_mRMR
cl = TreeBagger(100,X_train(:,ind(1:num)),Y_train);
out=str2double(predict(cl,X_test(:,ind(1:num))));

%check the accuracy
accuracy=sum(Y_test'==out)/length(Y_test)
 
%now only use the important features (optimal)
cl = TreeBagger(100,X_train(:,1:10),Y_train);
out=str2double(predict(cl,X_test(:,1:10)));

%check the accuracy
accuracy=sum(Y_test'==out)/length(Y_test)
