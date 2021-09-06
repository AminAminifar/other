function y=Mutate(x,mu)

    nVar=numel(x);
    
    nmu=ceil(mu*nVar);
    
    j=randsample(nVar,nmu);
    
    for i = j
        x(i) = randi(5);
    end
    y = x;
end