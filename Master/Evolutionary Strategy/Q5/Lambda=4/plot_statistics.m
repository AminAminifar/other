function [] = plot_statistics(stat, stopeval, runs)
  for i = 1 : runs
    histf(:,i) = stat(i).histf(1:stopeval);
  end
  plot(mean(histf, 2))
  xlabel('evaluations')
  ylabel('fitness')
end

