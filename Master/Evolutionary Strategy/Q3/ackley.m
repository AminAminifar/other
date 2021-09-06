function f = ackley(x)
  [ps,n] = size(x);
  f = 20 - 20 * exp(-0.2 * sqrt(sum(x .^ 2, 2) / n)) ...
        - exp(sum(cos(2 * pi * x), 2) / n) + exp(1);
end

