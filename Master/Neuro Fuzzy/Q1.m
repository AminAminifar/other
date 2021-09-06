clc
clear all
% data
num_data = 500;
x_in = unifrnd(-2,2,1,num_data);
f_out = zeros(1,num_data);
num_train_data = num_data - floor(.3*num_data);
num_test_data = floor(.3*num_data) - floor(.1*num_data);
num_eval_data = floor(.1*num_data);
for i=1:num_data
    f_out(i) = sinc(x_in(i));
end
% Number  of x and y membership functions
num_x = 15;
% initialization of sigmas and centers of membership functions
sig_x = (4/num_x)*rand(1,num_x);
c_x = unifrnd(-2,2,1,num_x);

% parameters
eta=0.05;
epoch= 150;
% initialization of p,q,r
p = rand(1,num_x);
r = rand(1,num_x);
% initialization of o1_x o1_y o2 o3 o4 o5 f
% % o1_x = zeros(1,num_x);
% % o1_y = zeros(1,num_y);
o2 = zeros(1,num_x);
o3 = zeros(1,num_x);
o4 = zeros(1,num_x);
% % o5 = zeros(1,num_x*num_y);
f = zeros(1,num_x);
o1_x = zeros(1,num_x);
error = zeros(1,epoch);


%% Run feed-forward and back propagation iteratively
for iteration=1:epoch
    for data_num=1:num_train_data
        x = x_in(data_num);
        d = f_out(data_num);
        % feed forward
        % x and y are inputs
        % 2 membership func for each input
        for i=1:num_x
            o1_x(i) = gaussmf(x,[sig_x(i) c_x(i)]);
        end
        % t-norm (o2=w)
        o2 = o1_x;
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        % %     clear sigma;
        % Layer 4
        for i=1:num_x
            f(i) = p(i)*x + r(i);
            o4(i) = o3(i)*f(i);
        end
        % Layer 5
        o5 = sum(o4);
        %%
        
        % feed-back
        e = d - o5;
        gr_E_p = e*-1*1*o3*x;
        gr_E_r = e*-1*1*o3*1;
        gr_E_sig_x = e*-1*1*f*(1/sigma)*1.*(o1_x.*((x-c_x).^2./(sig_x.^3)));
        gr_E_c_x = e*-1*1*f*(1/sigma)*1.*(o1_x.*((x-c_x)./(sig_x.^2)));        
        % Upadate parameters...
        p = p - eta*gr_E_p;
        r = r - eta*gr_E_r;
        sig_x = sig_x - eta*gr_E_sig_x;
        c_x = c_x - eta*gr_E_c_x;
        
    end
    %% Test
    summation = 0;
    for data_num = 1:num_eval_data
        x = x_in(data_num);
        d = f_out(data_num);
        % feed forward
                for i=1:num_x
            o1_x(i) = gaussmf(x,[sig_x(i) c_x(i)]);
        end
        % t-norm (o2=w)
        o2 = o1_x;
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        % %     clear sigma;
        % Layer 4
        for i=1:num_x
            f(i) = p(i)*x + r(i);
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