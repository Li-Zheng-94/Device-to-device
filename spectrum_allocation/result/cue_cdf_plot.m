clc
clear

n=50000;%要读入的行数
 
fid=fopen('cue_sinr_random.txt'); 
for i=1:n
    cue_sinr_random(i)=str2double(fgets(fid));
end 
fclose(fid);

fid=fopen('cue_sinr_rl.txt'); 
for i=1:n
    cue_sinr_rl(i)=str2double(fgets(fid));
end 
fclose(fid);

h1=cdfplot(cue_sinr_random);% 在matlab中画图我们使用cdfplot,这个命令
set(h1,'color','b','LineWidth',2);

hold on;
h2=cdfplot(cue_sinr_rl);% 在matlab中画图我们使用cdfplot,这个命令
set(h2,'color','r','LineWidth',2);

legend('random','reinforcement learning','Location','southeast');

xlabel('SINR');
ylabel('CDF');
title('CUE SINR CDF')
