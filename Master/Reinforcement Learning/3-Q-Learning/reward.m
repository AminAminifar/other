function r = reward( x,y,g,last_d,h,a )
%REWARD Summary of this function goes here
%   Detailed explanation goes here
r = 0;



if last_d==1
if h == 1%H1
    if x==3 && y==4
        r = -10;
    end
elseif h == 2%H2
    if x==2 && y==2
        r = -10;
    end
elseif h == 3%H3
    if x==4 && y==2
        r = -10;
    end
end
end

if  a==4%Left
    if x==1
        r = -1;
    elseif x==2 && (y==4 || y==5)
        r = -1;
    elseif x==3 && y==5
        r = -1;
    end
elseif a==3%Down
    if y==1
        r = -1;
    end
elseif a==2%Right
    if x==5
        r = -1;
    elseif x==1 && (y==4 || y==5)
        r = -1;
    elseif x==2 && y==5
        r = -1;
    end
elseif a==1%Up
    if y==5
        r = -1;
    end
end

if g == 1%G0
    if x==1 && y==5
        r = 10;
    end
elseif g == 2%G1
    if x==5 && y==5
        r = 10;
    end
elseif g == 3%G2
    if x==1 && y==1
        r = 10;
    end
elseif g == 4%G3
    if x==5 && y==1
        r = 10;
    end
end

end

