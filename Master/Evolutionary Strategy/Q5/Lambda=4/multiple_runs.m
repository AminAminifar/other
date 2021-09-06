function [] = multiple_runs(fitnessfct, n, lb, ub, stopeval, runs)
for i = 1 : runs
    [xopt, fopt, stat_1(i)] = es1(fitnessfct, n, lb, ub, stopeval);
end
for i = 1 : runs
    [xopt, fopt, stat_2(i)] = es2(fitnessfct, n, lb, ub, stopeval);
end
plot_statistics(stat_1, stopeval, runs)
hold on
plot_statistics(stat_2, stopeval, runs)
legend('(1+lambda)-ES','(1,lambda)-ES')
save('statistics.mat', 'stat_1', 'stat_2')
end

