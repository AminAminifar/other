clc
%%
in = zeros(0,100);
out = zeros(0,100);
j = 1;
for i=0:0.02:2
    in(j) = i;
    out(j) = humps(i);
    j = j+1;
end
%%

max_in = max(in);
max_out = max(out);

% if max_in > 1
%     in = in/max_in;
% end
% if max_out > 1
%     out = out/max_out;
% end
%%
data = randperm(100);
data_train = data(1: 70);
data_test1 = data(71: 80);
data_test2 = data(81: 90);
data_test3 = data(91: 100);
%%
% epsilon_test1 = 0.001;
epoch = 600;
max_n = 10;
err_2 = zeros(1,max_n - 2);

%%

%%
eta=0.005;
alpha=0.5;
betha=0.5;

n1=6;
%%

w1u=rand(1,n1);
w1l=rand(1,n1);
w2=rand(1,n1);

w1u_plot=zeros(epoch,n1);
w1l_plot=zeros(epoch,n1);
w2_plot=zeros(epoch,n1);
for j = 1:epoch
    for i = 1:length(data_train)
        d = out(data_train(i));
        x= in(data_train(i));
        
        net1u = w1u*x';
        net1l = w1l*x';

        
        o1u =max(tansig(net1u),tansig(net1l));
        o1l = min(tansig(net1u),tansig(net1l));
        
        o1=(o1u+o1l)/2;
        
        net2=w2*o1';
        o2=net2;
        
        e = d - o2;
        
        au = tansig(net1u);
        dau_dn = dtansig(net1u,au);
        
        al = tansig(net1l);
        dal_dn = dtansig(net1l,al);
        
        
        
        gr_E_w2 = -1*e*1*o1;
        gr_E_w1u = -1*e*w2*0.5.*dau_dn*x;
        gr_E_w1l = -1*e*w2*0.5.*dal_dn*x;
        
        w2 = w2 - eta*gr_E_w2;
        w1u = w1u - eta*gr_E_w1u;
        w1l = w1l - eta*gr_E_w1l;
        
        
    end
    w2_plot(j,:)=w2;
    w1u_plot(j,:)=w1u;
    w1l_plot(j,:)=w1l;
end
j=1:epoch;
figure('Name','weight2','NumberTitle','off');
plot(j,w2_plot(1:epoch,1));
figure('Name','weight1_upper_bound','NumberTitle','off');
plot(j,w1u_plot(1:epoch,1));
figure('Name','weight1_lower_bound','NumberTitle','off');
plot(j,w1l_plot(1:epoch,1));

figure('Name','realVSestimation','NumberTitle','off');
for i = 1:length(in)
    
    x= in(i);
    
    net1u = w1u*x';
    net1l = w1l*x';
    
    o1u =max(tansig(net1u),tansig(net1l));
    o1l = min(tansig(net1u),tansig(net1l));
    
    o1=(o1u+o1l)/2;
    
    net2=w2*o1';
    o2=net2;
    
    plot (in(i),o2)
    hold on
end
plot(in,out,'-')
hold off

%%


