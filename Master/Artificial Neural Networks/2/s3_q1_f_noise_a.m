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
% data_test2=awgn(data_test2,10,'measured');
%%
epsilon_test1 = 0.001;
epoch_test1 = 450;
max_n = 10;
err_2 = zeros(1,max_n - 2);

%%

%%
eta=0.005;
for n1=3:10
    %%
    c_f=zeros(epoch_test1,n1);
    sig_f=zeros(epoch_test1,n1);
    w_f=zeros(epoch_test1,n1);
    for n = 1 :10
        E_test2 = zeros(1, 10);
        c =rand(1,n1);
        sig=rand(1,n1);
        w=rand(1,n1);
        
        E_test1 = zeros(1 ,length(data_test1));
        
        mean_test1_err = 1000;
        
        
        %%
        
        c_f_1=zeros(epoch_test1,n1);
        sig_f_1=zeros(epoch_test1,n1);
        w_f_1=zeros(epoch_test1,n1);
        for j = 1:epoch_test1
            if mean_test1_err < epsilon_test1
                break;
            end
            %%
            for i = 1:length(data_train)
                d = out(data_train(i));
                net1 = abs(in(data_train(i))-c);
                
                o1 = exp(-0.5*((net1./sig).^2));
                
                net2 = w*o1';
                o2 = net2;
                e = d - o2;
                
                %             a = tansig(net1);
                %             da_dn = dtansig(net1,a);
                
                gr_E_w = -1*(e)*1*o1;
                gr_E_c=-1*e*w.*((in(data_train(i))-c)./(sig.^2)).*o1;
                gr_E_sig=-1*e*w.*(((in(data_train(i))-c).^2)./(sig.^3)).*o1;
                
                w = w - eta*gr_E_w;
                c= c- eta*gr_E_c;
                sig=sig- eta*gr_E_sig;
                
            end
            
            c_f_1(j,:)=c;
            sig_f_1(j,:)=sig;
            w_f_1(j,:)=w;
            global_j_f_1=j;
            
            %%
            for i = 1:length(data_test1)
                d_test1 = out(data_test1(i));
                i_test1 = in(data_test1(i));
                net1_test1 = abs(i_test1-c);
                o1_test1 = exp(-0.5*((net1_test1./sig).^2));
                net2_test1 = w * o1_test1';
                o2_test1 = net2_test1;
                
                err_test1 = d_test1 - o2_test1;
                E_test1(i) = E_test1(i) + 1/2*(err_test1^2);
            end
            %%
            mean_test1_err = mean(E_test1);
            E_test1 = zeros(1 ,10);
        end
        %%
        %%
        for i = 1:length(data_test2)
            d_test2 = out(data_test2(i));
            i_test2 = in(data_test2(i));
            net1_test2 = abs(i_test2-c);
            o1_test2 = exp(-0.5*((net1_test2./sig).^2));
            net2_test2 = w * o1_test2';
            o2_test2 = net2_test2;
            
            err_test2 = d_test2 - o2_test2;
            E_test2(i) = E_test2(i) + 1/2*(err_test2^2);
            
        end
        mean_test2_err(n) = mean(E_test2);
        %%
        if  n==1
            c_f=c_f_1;
            sig_f=sig_f_1;
            w_f=w_f_1;
            global_j_f=global_j_f_1;
        else
            if mean_test2_err(n)<mean_test2_err(n-1)
                c_f=c_f_1;
                sig_f=sig_f_1;
                w_f=w_f_1;
                global_j_f=global_j_f_1;
            end
        end
        %%
    end
    %%
    mean_test2_err_diff_n1(n1)=mean( mean_test2_err)
    if (n1==3)
        c_best=c_f;
        sig_best=sig_f;
        w_best=w_f;
        global_j_best=global_j_f;
        n_best=n1;
    else
        if(mean_test2_err_diff_n1(n1)<mean_test2_err_diff_n1(n1-1))
            c_best=c_f;
            sig_best=sig_f;
            w_best=w_f;
            global_j_best=global_j_f;
            n_best=n1;
        end
    end
end
%%
% data_test1_noisey=awgn(data_test1,10,'measured');
% data_test1=data_test1_noisey;
for i = 1:length(data_test1)
    d_test1 = out(data_test1(i));
    i_test1 = awgn(in(data_test1(i)),10,'measured');%in(data_test1(i));
    net1_test1 = abs(i_test1-c);
    o1_test1 = exp(-0.5*((net1_test1./sig).^2));
    net2_test1 = w * o1_test1';
    o2_test1 = net2_test1;
    
    err_test1 = d_test1 - o2_test1
    E_test1(i) = E_test1(i) + 1/2*(err_test1^2)
end
error= mean( E_test1)

n_best
j=1:global_j_best;
figure('Name','center_point','NumberTitle','off');
plot(j,c_best(1:global_j_best,1));
figure('Name','sigma','NumberTitle','off');
plot(j,sig_best(1:global_j_best,1));
figure('Name','weight','NumberTitle','off');
plot(j,w_best(1:global_j_best,1));
% n1=3:10
% plot(n1, mean_test2_err_diff_n1)
figure('Name','Real VS Estimation','NumberTitle','off');
for i = 1:length(in)
    
    net1 = abs(in(i)-c_best(global_j_best,1:n_best));
    
    o1 = exp(-0.5*((net1./sig_best(global_j_best,1:n_best)).^2));
    
    net2 = w_best(global_j_best,1:n_best)*o1';
    o2= net2;
    
    plot (in(i),o2)
    hold on
end
plot(in,out,'-')
% hold on
% plot (in,o2)
hold off

