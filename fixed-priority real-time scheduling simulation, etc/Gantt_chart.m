function [ output_args ] = Gantt_chart( timeline )
%GANTT_CHART Summary of this function goes here
%   Detailed explanation goes here

n=numel(timeline(1,:));
Max_time=timeline(1,n);
Max_task=max(timeline(2,:));
x=0:.01:Max_time;
col=hsv(Max_task+1);

figure;
plot(x,1);
hold on
for i=2:Max_task
    plot(x,i);
end
for i=1:n-1
    plot(timeline(1,i:i+1),[timeline(2,i),timeline(2,i)],'color',col(timeline(2,i)+1,:),'LineWidth',20);
end
hold off

title('Gantt Chart');
xlabel('Time');
ylabel('Tasks');
axis([0 Max_time 0 Max_task+1]);

end

