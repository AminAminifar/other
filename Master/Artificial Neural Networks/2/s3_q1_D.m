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

w1=rand(1,n1);
w2=rand(1,n1);
au=rand(1,n1);
al=rand(1,n1);
bu=rand(1,n1);
bl=rand(1,n1);

w2_plot =zeros(epoch,n1);
w1_plot =zeros(epoch,n1);
au_plot =zeros(epoch,n1);
al_plot =zeros(epoch,n1);
bu_plot =zeros(epoch,n1);
bl_plot =zeros(epoch,n1);
for j = 1:epoch
    for i = 1:length(data_train)
        d = out(data_train(i));
        x= in(data_train(i));
        
        net1 = w1*x';
        
        
        
        o1u = max(au.*tanh(bu.*net1),al.*tanh(bl.*net1));
        o1l = min(au.*tanh(bu.*net1),al.*tanh(bl.*net1));
        
        o1=(o1u+o1l)/2;
        
        net2=w2*o1';
        o2=net2;
        
        e = d - o2;
        
        
        
        
        gr_E_w2 = -1*e*1*o1;
        gr_E_w1 =e*(-1)*1*w2*0.5*(-1).*au.*bu.*(tanh(bu.*net1).^2-1)*x+e*(-1)*1*w2*0.5*(-1).*al.*bl.*(tanh(bl.*net1).^2-1)*x;
        gr_E_au =e*(-1)*1*w2*0.5.*tanh(bu.*net1);
        gr_E_al =e*(-1)*1*w2*0.5.*tanh(bl.*net1);
        gr_E_bu =e*(-1)*1*w2*0.5*(-1).*au.*net1.*(tanh(bu.*net1).^2-1);
        gr_E_bl =e*(-1)*1*w2*0.5*(-1).*al.*net1.*(tanh(bl.*net1).^2-1);
        
        w2 = w2 - eta*gr_E_w2;
        w1 = w1 - eta*gr_E_w1;
        au = au - eta*gr_E_au;
        al = al - eta*gr_E_al;
        bu = bu - eta*gr_E_bu;
        bl = bl - eta*gr_E_bl;
        
        
    end
    w2_plot(j,:)=w2;
    w1_plot(j,:)=w1;
    au_plot(j,:)=au;
    al_plot(j,:)=al;
    bu_plot(j,:)=bu;
    bl_plot(j,:)=bl;
end
j=1:epoch;
figure('Name','weight1','NumberTitle','off');
plot(j,w1_plot(1:epoch,1));
figure('Name','weight2','NumberTitle','off');
plot(j,w2_plot(1:epoch,1));
figure('Name','a','NumberTitle','off');
plot(j,au_plot(1:epoch,1));
hold on
plot(j,al_plot(1:epoch,1));
hold off
figure('Name','b','NumberTitle','off');
plot(j,bu_plot(1:epoch,1));
hold on
plot(j,bl_plot(1:epoch,1));
hold off

figure('Name','realVSestimation','NumberTitle','off');
for i = 1:length(in)
    
    x= in(i);
    
    net1 = w1*x';

    o1u = max(au.*tanh(bu.*net1),al.*tanh(bl.*net1));
    o1l = min(au.*tanh(bu.*net1),al.*tanh(bl.*net1));
    
    o1=(o1u+o1l)/2;
    
    net2=w2*o1';
    o2=net2;

    
    plot (in(i),o2)
    hold on
end
plot(in,out,'-')
hold off

%%


