clc;
clear all;
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

if max_in > 1
    in = in/max_in;
end
if max_out > 1
    out = out/max_out;
end
%%
data = randperm(100);
data_train = data(1: 70);
data_test1 = data(71: 80);
data_test2 = data(81: 90);
data_test3 = data(91: 100);
%%
% epsilon_test1 = 0.001;
epoch = 450;
max_n = 10;
err_2 = zeros(1,max_n - 2);

%%

%%
eta=0.005;
alpha=0.5;
betha=0.5;

n1=6;
%%
c =rand(1,n1);
sig=rand(1,n1);
wu=rand(1,n1);
wl=rand(1,n1);

wu_plot=zeros(epoch,n1);
wl_plot=zeros(epoch,n1);
c_plot=zeros(epoch,n1);
sig_plot=zeros(epoch,n1);
for j = 1:epoch
    for i = 1:length(data_train)
        d = out(data_train(i));
        net1 = abs(in(data_train(i))-c);
        
        o1 = exp(-0.5*((net1./sig).^2));
        
        net2u = wu*o1';
        net2l = wl*o1';
        
        o2u =max(net2u,net2l);
        o2l = min(net2u,net2l);
        
        o2=(o2u+o2l)/2;
        
        au = tansig(net2u);
        dau_dn = dtansig(net2u,au);
        
        al = tansig(net2l);
        dal_dn = dtansig(net2l,al);
        
        e = d - o2;
        
        
        gr_E_wu = -1*(e)*.5*dau_dn*o1;
        gr_E_wl =-1*(e)*.5*dal_dn*o1;
        gr_E_c=-1*e*dal_dn*wl.*((in(data_train(i))-c)./(sig.^2)).*o1-e*dau_dn*wu.*((in(data_train(i))-c)./(sig.^2)).*o1;
        gr_E_sig=-1*e*dal_dn*wl.*(((in(data_train(i))-c).^2)./(sig.^3)).*o1-e*dau_dn*wu.*(((in(data_train(i))-c).^2)./(sig.^3)).*o1;
        
        wu = wu - eta*gr_E_wu;
        wl = wl - eta*gr_E_wl;
        c= c- eta*gr_E_c;
        sig=sig- eta*gr_E_sig;
        
    end
    wu_plot(j,:)=wu;
    wl_plot(j,:)=wl;
    c_plot(j,:)=c;
    sig_plot(j,:)=sig;
end
j=1:epoch;
figure('Name','center_point','NumberTitle','off');
plot(j,c_plot(1:epoch,1));
figure('Name','sigma','NumberTitle','off');
plot(j,sig_plot(1:epoch,1));
figure('Name','weight_upper_bound','NumberTitle','off');
plot(j,wu_plot(1:epoch,1));
figure('Name','weight_lower_bound','NumberTitle','off');
plot(j,wl_plot(1:epoch,1));

for i = 1:length(in)
    
    net1 = abs(in(i)-c);
    
    o1 = exp(-0.5*((net1./sig).^2));
    
    net2u = wu*o1';
    net2l = wl*o1';
    
    o2u =max(net2u,net2l);
    o2l = min(net2u,net2l);
    
    o2=(o2u+o2l)/2;
    
    plot (in(i),o2)
    hold on
end
plot(in,out,'-')
hold off

%%


