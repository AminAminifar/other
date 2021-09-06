function [] = multiple_runs(fitnessfct, n, lb, ub, stopeval, runs)
  for i = 1 : runs
    [xopt, fopt, stat(i)] = es(fitnessfct, n, lb, ub, stopeval);
  end
  save('statistics.mat', 'stat')
end

