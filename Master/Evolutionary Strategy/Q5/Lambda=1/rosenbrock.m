function f = rosenbrock(x)
  f = 100 * sum((x(1:end-1).^2 - x(2:end)).^2) + sum((x(1:end-1)-1).^2);
end

