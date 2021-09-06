c=[1,1,1,6,2];
p=[14,18,18,112,14];
hp=[1,2,4,3,5];
n=5;
until=10;
timeline=simulation(c,p,hp,n,until);
Gantt_chart(timeline);