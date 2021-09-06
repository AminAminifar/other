function [ timeline ] = simulation(c,p,hp,quantity_of_tasks,until)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


job_execution_time=c;
job_release_time=p;
time=0;
current=hp(1);%assumption: all jobs released at time 0 at once
%local i j;
x=until;
next=current;
n=quantity_of_tasks;

for i=1:x
    
timeline(1,i)=time;
timeline(2,i)=current;

if next==0;%idle
    next_time=min(job_release_time);
else
    next_time=min((time+job_execution_time(current)),(min(job_release_time)));
    
    job_execution_time(current)=job_execution_time(current)-(next_time-time);%updating job_execution_time for current
end


%change if next_time>job....

%do job or idle

if next_time==min(job_release_time)
    
    for j=1:n %updating job_execution_time and job_release_time if job_release_time(j)==min(job_release_time)
        if job_release_time(j)==next_time
            job_execution_time(j)=c(j);
            job_release_time(j)=job_release_time(j)+p(j);
        end
    end
    
end
    
 next=0;%idle
    %next_time==time+job_execution_time(current)
    for j=n:-1:1 %changing next
              if  job_execution_time(hp(j))>0
                  next=hp(j);
              end
    end
  
    

time=next_time;
current=next;


end

