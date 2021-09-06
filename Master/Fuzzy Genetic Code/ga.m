clc;
close all;
clear all;
npop=300;

pop.pos=[];
pop.cost=[];

nvar=30;

popall=repmat(pop,[npop 1]);

pc=.7;
pm=.4;
rate_mutate=0.5;

for i=1:npop
    popall(i).pos=randperm(nvar);
    popall(i).cost=costfunction(popall(i).pos);   
end

num_mutete=ceil(npop*pm);
num_cross=ceil(npop*pc);

mpopall=repmat(pop,[num_mutete 1]);
cpopall=repmat(pop,[num_cross*2 1]);

maxiter=300;
ccc=zeros(maxiter,1);
for k=1:maxiter
    for j=1:num_mutete
        r1=randi(npop);
        
        mpopall(j).pos=Mutate(popall(r1).pos,rate_mutate);
        mpopall(j).cost=costfunction(mpopall(j).pos); 
    end
    b=1;
    for j=1:num_cross
        r1=randi(npop);
        r2=randi(npop);
        
        if rand <=.8
            % PMX Crossover
            zzz = PMX_Crossover(popall(r1).pos,popall(r2).pos);
        else
            % PX Crossover
            zzz = Permutatin_Crossover(popall(r1).pos,popall(r2).pos);
        end
        
        cpopall(b).pos=zzz(1,:);
        cpopall(b).cost=costfunction(cpopall(b).pos); 
        b=b+1;
        cpopall(b).pos=zzz(2,:);
        cpopall(b).cost=costfunction(cpopall(b).pos); 
        b=b+1;
    end
    
    temp_pop=[popall
              mpopall
              cpopall];
    
      [a,b]=sort([temp_pop.cost]);
      popall=temp_pop(b(1:npop));
    
      ccc(k)=popall(1).cost;

      figure(2);
      plot(ccc(1:k));
      
      if(popall(1).cost==0)
         rrr=popall(1).pos;
         break;
      end
end


