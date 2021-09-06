function [xp, fp, stat] = es2(fitnessfct, n, lb, ub, stopeval)
% [xp, fp, stat] = es(fitnessfct, n, lb, ub, stopeval)
% Strategy parameters
if strcmp(fitnessfct,'sphere')
    fitnessfct = @sphere;
elseif strcmp(fitnessfct,'ackley')
    fitnessfct = @ackley;
elseif strcmp(fitnessfct,'rosenbrock')
    fitnessfct = @rosenbrock;
end

% Initialize
popsize = 4;
for i=1:popsize
    x(i,:) = lb + (ub-lb)*rand(1,n);
    f(i) = fitnessfct(x(i,:));
end
[fp, index]= min(f);
xp = x(index,:);
clear index;
sigma = rand;
evalcount = 0;
k = 10;
sig_eval = zeros(1,k);

% Statistics administration
stat.name = '(1,lambda)-ES';
stat.evalcount = 0;
stat.histsigma = zeros(1, stopeval);
stat.histf = zeros(1, stopeval);

% Evolution cycle
while evalcount < stopeval
    %Selection
    [f_temp, i_temp]= min(f);
    % Generate offspring and evaluate
    xo = x(i_temp,:) + 10*normrnd(0,sigma,1,n);% generate offspring from parent xp
    fo = fitnessfct(xo); % evaluate xo using fitnessfct
    evalcount = evalcount + 1;
    clear i_temp;
    
    % select best and update success-rate and update stepsize
    % Important: MINIMIZATION!
    %Selection
    
    if f_temp>fo
        if mod(evalcount,k)~=0
            sig_eval(mod(evalcount,k)) = 1;
        else
            sig_eval(k) = 1;
        end
    end
    if mod(evalcount,k)==0
        if sum(sig_eval)/k<1/5
            sigma = sigma*.8;
        elseif sum(sig_eval)/k>1/5
            sigma = sigma/.8;
        end
        sig_eval = zeros(1,k);
    end
    [~, i_temp]= max(f);
%     if f_temp>fo
        x(i_temp,:) = xo;
        f(i_temp) = fo;
%     end
    clear f_temp i_temp;
    
    
    [fp, index]= min(f);
    xp = x(index,:);
    clear index;
    % Statistics administration
    stat.histsigma(evalcount) = sigma;% stepsize history
    stat.histf(evalcount) = fp;% fitness history
    
    % if desired: plot the statistics
end
xp
fp
end

