clc
clear all
% data
num_data = 1000;
x_in = unifrnd(-2,2,1,num_data);
y_in = unifrnd(-2,2,1,num_data);
f_out = zeros(1,num_data);
num_train_data = num_data - floor(.3*num_data);
num_test_data = floor(.3*num_data) - floor(.1*num_data);
num_eval_data = floor(.1*num_data);
for i=1:num_data
    f_out(i) = x_in(i)^2 + y_in(i)^2;
end
% Number  of x and y membership functions
num_x = 10;
num_y = 10;
% initialization of sigmas and centers of membership functions
sig_x = (4/num_x)*rand(1,num_x);
c_x = unifrnd(-2,2,1,num_x);
sig_y = (4/num_y)*rand(1,num_y);
c_y = unifrnd(-2,2,1,num_y);

for i=1:num_x
    t_sig_x(((i-1)*num_y)+1:(i)*num_y) = sig_x(i)*ones(1,num_y);
    t_c_x(((i-1)*num_y)+1:(i)*num_y) = c_x(i)*ones(1,num_y);
end
t_sig_y = sig_y*repmat(eye(num_y),1,num_x);
t_c_y = c_y*repmat(eye(num_y),1,num_x);
% parameters
eta=0.05;
epoch= 150;
% initialization of p,q,r
p = rand(1,num_x*num_y);
q = rand(1,num_x*num_y);
r = rand(1,num_x*num_y);
% initialization of o1_x o1_y o2 o3 o4 o5 f
% % o1_x = zeros(1,num_x);
% % o1_y = zeros(1,num_y);
o2 = zeros(1,num_x*num_y);
o3 = zeros(1,num_x*num_y);
o4 = zeros(1,num_x*num_y);
% % o5 = zeros(1,num_x*num_y);
f = zeros(1,num_x*num_y);
t_o1_x = zeros(1,num_x*num_y);
t_o1_y = zeros(1,num_x*num_y);
error = zeros(1,epoch);


%% Run feed-forward and back propagation iteratively
for iteration=1:epoch
    for data_num=1:num_train_data
        x = x_in(data_num);
        y = y_in(data_num);
        d = f_out(data_num);
        % feed forward
        % x and y are inputs
        % 2 membership func for each input
        for i=1:num_x*num_y
            t_o1_x(i) = gaussmf(x,[t_sig_x(i) t_c_x(i)]);
        end
        for i=1:num_x*num_y
            t_o1_y(i) = gaussmf(y,[t_sig_y(i) t_c_y(i)]);
        end
        % t-norm (o2=w)
        o2 = t_o1_x.*t_o1_y;
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        % %     clear sigma;
        % Layer 4
        for i=1:num_x*num_y
            f(i) = p(i)*x + q(i)*y + r(i);
            o4(i) = o3(i)*f(i);
        end
        % Layer 5
        o5 = sum(o4);
        %%
        
        % feed-back
        e = d - o5;
        gr_E_p = e*-1*1*o3*x;
        gr_E_q = e*-1*1*o3*y;
        gr_E_r = e*-1*1*o3*1;
        gr_E_sig_x = e*-1*1*f*(1/sigma).*t_o1_y.*(t_o1_x.*((x-t_c_x).^2./(t_sig_x.^3)));
        gr_E_c_x = e*-1*1*f*(1/sigma).*t_o1_y.*(t_o1_x.*((x-t_c_x)./(t_sig_x.^2)));
        gr_E_sig_y = e*-1*1*f*(1/sigma).*t_o1_x.*(t_o1_y.*((y-t_c_y).^2./(t_sig_y.^3)));
        gr_E_c_y = e*-1*1*f*(1/sigma).*t_o1_x.*(t_o1_y.*((y-t_c_y)./(t_sig_y.^2)));
        
        % Upadate parameters...
        p = p - eta*gr_E_p;
        q = q - eta*gr_E_q;
        r = r - eta*gr_E_r;
        t_sig_x = t_sig_x - eta*gr_E_sig_x;
        t_c_x = t_c_x - eta*gr_E_c_x;
        t_sig_y = t_sig_y - eta*gr_E_sig_y;
        t_c_y = t_c_y - eta*gr_E_c_y;
        
    end
    %% Test
    summation = 0;
    for data_num = 1:num_test_data
        x = x_in(data_num);
        y = y_in(data_num);
        d = f_out(data_num);
        % feed forward
        for i=1:num_x*num_y
            t_o1_x(i) = gaussmf(x,[t_sig_x(i) t_c_x(i)]);
        end
        for i=1:num_x*num_y
            t_o1_y(i) = gaussmf(y,[t_sig_y(i) t_c_y(i)]);
        end
        % t-norm (o2=w)
        o2 = t_o1_x.*t_o1_y;
        %Normalize
        sigma = sum(o2);
        for i=1:length(o2)
            o3(i) = o2(i)/sigma;
        end
        clear sigma;
        % Layer 4
        for i=1:num_x*num_y
            f(i) = p(i)*x + q(i)*y + r(i);
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