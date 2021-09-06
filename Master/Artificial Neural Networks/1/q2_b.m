clc;
clear all;
in = zeros(0,100);
out = zeros(0,100);
j = 1;
for i=0:0.02:2
    in(j) = i;
    out(j) = humps(i);
    j = j+1;
end

max_in = max(in);
max_out = max(out);

if max_in > 1
    in = in/max_in;
end
if max_out > 1
    out = out/max_out;
end

a = 1.7159;
b = 2/3;

data = randperm(100);
data_train = data(1: 70);
data_test1 = data(71: 80);
data_test2 = data(81: 90);
data_test3 = data(91: 100);

epsilon_test1 = 0.01;
epoch_tets1 = 250;
max_n = 10;
err_2 = zeros(1,max_n - 2);
figure('Name','1','NumberTitle','off');
for best_num_n = 3 : max_n
    w1 = rand(1,best_num_n);
    w2 = rand(1,best_num_n);
    
    net1 = zeros(1,best_num_n);
    net2 = 0;
    
    o1 = zeros(1,best_num_n);
    o2 = 0;
    
    eta = 0.25;
    E_tets1 = zeros(1 ,length(data_test1));
    
    mean_test1_err = 1000;
    for j = 1:epoch_tets1
        if mean_test1_err < epsilon_test1
            break;
        end
        for i = 1:length(data_train)
            d = out(data_train(i));
            net1 = in(data_train(i)) * w1;
            o1 = a*tanh(b*net1);
            net2 = w2*o1';
            o2 = net2;
            e = d - o2;
            
            fnet = a*tanh(b*net1);
            dfnet_dn = sech(fnet).*sech(fnet);
            
            gr_E_w2 = -1*(e)*1*o1;
            gr_E_w1 = -1*e*1*w2.*dfnet_dn*in(data_train(i));
            
            w1 = w1 - eta*gr_E_w1;
            w2 = w2 - eta*gr_E_w2;
        end
        
        for i = 1:length(data_test1)
            d_test1 = out(data_test1(i));
            i_test1 = in(data_test1(i));
            net1_test1 = i_test1*w1;
            o1_test1 = a*tanh(b*net1_test1);
            net2_test1 = w2 * o1_test1';
            o2_test1 = net2_test1;
            
            e_test1 = d_test1 - o2_test1;
            E_tets1(i) = E_tets1(i) + 1/2*(e_test1^2);
        end
        mean_test1_err = mean(E_tets1);
        E_tets1 = zeros(1 ,10);
    end
    
    
    mean_test2 = zeros(1,10);
    for n = 1 :10
        E_test2 = zeros(1, 10);
        for k = 1:length(data_test2)
            d_test2 = out(data_test2(k));
            i_test2 = in(data_test2(k));
            net1_test2 = i_test2*w1;
            o1_test2 = a*tanh(b*net1_test2);
            net2_test2 = w2 * o1_test2';
            o2_test2 = net2_test2;
            
            e_test2 = d_test2 - o2_test2;
            E_test2(k) = 1/2*(e_test2^2);
        end
        mean_test2(n) = mean(E_test2);
    end
    err_2(best_num_n) = mean(mean_test2);
    

    if(best_num_n == 3)
        bst_w1 = w1;
        bst_w2 = w2;
        ne = best_num_n;
    elseif(err_2(best_num_n) < err_2(ne))
        bst_w1 = w1;
        bst_w2 = w2;
        ne = best_num_n;
    end
    
    
    plot(best_num_n,err_2(best_num_n),'o')
    hold on;
    best_num_n;
    ne
end
hold off

E_test1_bst = zeros(1,length(data_test1));
for i = 1:length(data_test1)
    d_test1_bst = out(data_test1(i));
    i_test1_bst = in(data_test1(i));
    net1_test1_bst = i_test1_bst*bst_w1;
    o1_test1_bst = a*tanh(b*net1_test1_bst);
    net2_test1_bst = bst_w2 * o1_test1_bst';
    o2_test1_bst = net2_test1_bst;
    
    e_test1_bst = d_test1_bst - o2_test1_bst;
    E_test1_bst(i) = E_test1_bst(i) + 1/2*(e_test1_bst^2);
end
i = 1:length(data_test1);
figure('Name','first test error for opt neuron','NumberTitle','off');
plot(i,E_test1_bst,'-s','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',10);

E_test3_bst = zeros(1,length(data_test3));
for i = 1:length(data_test3)
    d_test3_bst = out(data_test3(i));
    i_test3_bst = in(data_test3(i));
    net1_test3_bst = i_test3_bst*bst_w1;
    o1_test3_bst = a*tanh(b*net1_test3_bst);
    net2_test3_bst = bst_w2 * o1_test3_bst';
    o2_test3_bst = net2_test3_bst;
    
    e_test3_bst = d_test3_bst - o2_test3_bst;
    E_test3_bst(i) = E_test3_bst(i) + 1/2*(e_test3_bst^2);
end
i = 1:length(data_test3);
figure('Name','final error for opt neuron','NumberTitle','off');
plot(i,E_test3_bst,'-s','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',10);





