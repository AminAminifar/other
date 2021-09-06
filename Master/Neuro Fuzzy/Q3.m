clc 
clear all
% data
data = load('Data_mackey.mat');
data = struct2cell(data);
data = cell2mat(data);
num_data = 1000;
x_in = zeros(num_data,5);
for i=1:4
    x_in(:,i) = data(:,i);
end
num_train_data = num_data - floor(.3*num_data);
num_test_data = floor(.3*num_data) - floor(.1*num_data);
num_eval_data = floor(.1*num_data);
f_out = data(:,5);
% Number  of x and y membership functions
num_x = 4*ones(1,4);
% initialization of sigmas and centers of membership functions
% sig_x = zeros(4,);
% c_x = zeros(4,);
for i=1:4
    sig_x(i,:) = (max(data(:,i))-min(data(:,i))/num_x(i))*rand(1,num_x(i));
    c_x(i,:) = unifrnd(min(data(:,i)),max(data(:,i)),1,num_x(i));
end
num_rules = num_x(1)*num_x(2)*num_x(3)*num_x(4);
t_sig_x = zeros(4,num_rules);
t_c_x = zeros(4,num_rules);
index = 1;
for i1=1:length(sig_x(1,:))
    for i2=1:length(sig_x(2,:))
        for i3=1:length(sig_x(3,:))
            for i4=1:length(sig_x(4,:))
                t_sig_x(1,index) = sig_x(1,i1);
                t_c_x(1,index) = c_x(1,i1);
                t_sig_x(2,index) = sig_x(2,i2);
                t_c_x(2,index) = c_x(2,i2);
                t_sig_x(3,index) = sig_x(3,i3);
                t_c_x(3,index) = c_x(3,i3);
                t_sig_x(4,index) = sig_x(4,i4);
                t_c_x(4,index) = c_x(4,i4);
                index = index + 1;
            end
        end
    end
end
clear index;
% parameters
eta=0.05;
epoch= 10;
% initialization of p,q,r
p = rand(4,num_rules);
r = rand(1,num_rules);
% initialization of o1_x o1_y o2 o3 o4 o5 f
% % o1_x = zeros(1,num_x);
% % o1_y = zeros(1,num_y);
o2 = zeros(1,num_rules);
o3 = zeros(1,num_rules);
o4 = zeros(1,num_rules);
% % o5 = zeros(1,num_x*num_y);
f = zeros(1,num_rules);
t_o1_x = zeros(4,num_rules);
error = zeros(1,epoch);


%% Run feed-forward and back propagation iteratively
for iteration=1:epoch
    for data_num=1:num_train_data
        x = x_in(data_num,1:4);
        d = f_out(data_num);
        % feed forward
        % x and y are inputs
        % 2 membership func for each input
        for j=1:4
            for i=1:num_rules
                t_o1_x(j,i) = gaussmf(x(j),[t_sig_x(j,i) t_c_x(j,i)]);
            end
        end
        % t-norm (o2=w)
        o2 = t_o1_x(1,:).*t_o1_x(2,:).*t_o1_x(3,:).*t_o1_x(4,:);
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        % %     clear sigma;
        % Layer 4
        for i=1:num_rules
            f(i) = x*p(:,i) + r(i);
            o4(i) = o3(i)*f(i);
        end
        % Layer 5
        o5 = sum(o4);
        %%
        
        % feed-back
        e = d - o5;
        gr_E_p = e*-1*1*x'*o3;
        gr_E_r = e*-1*1*o3*1;
        for i=1:4
            gr_E_sig_x(i,:) = e*-1*1*f*(1/sigma).*o2.*((x(i)-t_c_x(i,:)).^2./(t_sig_x(i,:).^3));
            gr_E_c_x(i,:) = e*-1*1*f*(1/sigma).*o2.*((x(i)-t_c_x(i,:))./(t_sig_x(i,:).^2));
        end
        
        % Upadate parameters...
        p = p - eta*gr_E_p;
        r = r - eta*gr_E_r;
        t_sig_x = t_sig_x - eta*gr_E_sig_x;
        t_c_x = t_c_x - eta*gr_E_c_x;
    end
    %% Test
    summation = 0;
    for data_num = 1:num_test_data
        x = x_in(data_num,1:4);
        d = f_out(data_num);
        % feed forward
        for j=1:4
            for i=1:num_rules
                t_o1_x(j,i) = gaussmf(x(j),[t_sig_x(j,i) t_c_x(j,i)]);
            end
        end
        % t-norm (o2=w)
        o2 = t_o1_x(1,:).*t_o1_x(2,:).*t_o1_x(3,:).*t_o1_x(4,:);
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        % %     clear sigma;
        % Layer 4
        for i=1:num_rules
            f(i) = x*p(:,i) + r(i);
            o4(i) = o3(i)*f(i);
        end
        % Layer 5
        o5 = sum(o4);
        
        %error
        summation = summation + (d-o5)^2;
    end
    error(iteration) = .5*summation;
    plot(error(1:iteration))
    pause(.005)
end