function [ response_time ] = responsetime_analysis( c,p,hp,i)

%   Detailed explanation goes here
    k=2;
    r(1)=c(i);
    
    while true
        j=1;
        sigma=0;
        while hp(j)~=i 
        sigma=sigma+(ceil(r(k-1)/p(hp(j)))*c(hp(j)));
        j=j+1;
        end
        r(k)=c(i)+sigma;
        
        
     if r(k)>p(i)
         r(k)=inf;
         k=k+1;
         break; 
     elseif r(k)==r(k-1)
         k=k+1;
         break;
     end
     k=k+1;
    end
    response_time=r(k-1)
end

