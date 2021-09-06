function r = reward( x,phi )
%REWARD Summary of this function goes here
%   Detailed explanation goes here

if ((x>=10-.1)&(x<=10+.1)&...
        (phi>=90-1)&(phi<=90+1))
    r = 0;
    % elseif x<0
    %     r = -1;
elseif x>15
    r = -1.2;
elseif x<5
    r = -1.2;
elseif x>18
    r = -1.4;
elseif x<2
    r = -1.4;
    % elseif x>0 && x<5
    %     r = -10;
    % elseif phi<45 || phi>135
    %     r = -10;
else
    r = -1;
end
end

