function [ output ] = Infererece(x, chromosome, c, sigma  )

    s_mf = -inf;
    chromosome = reshape(chromosome,125,4);
    n = size(chromosome,1);
    for i = 1:n
        chr = chromosome(i,:);
        
        o1 = gaussmf(x(1),[sigma c(chr(1))]);
        o2 = gaussmf(x(2),[sigma c(chr(2))]);
        o3 = gaussmf(x(3),[sigma c(chr(3))]);
        
        FP1 = o1*o2*o3; % t-norm
        
        support = -1:0.1:1;
        mf = gaussmf(support,[sigma c(chr(4))]);
        mf = mf * FP1; % t-norm
        
        s_mf = max(s_mf, mf);
    end
    output = defuzz(support,s_mf,'centroid');
end

