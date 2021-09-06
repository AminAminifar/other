function [ nx,ny,ng,nd ] = new_state( x,y,g,d,a,h )
%NEW_STATE Summary of this function goes here
%   Detailed explanation goes here
[ nx,ny,ng,nd ] = deal( x,y,g,d );

if a==1%Up
    ny = y + 1;
    if y==5
        ny = y;
    end
elseif a==2%Right
    nx = x + 1;
    if x==5
        nx = x;
    elseif x==1 && (y==4 || y==5)
        nx = x;
    elseif x==2 && y==5
        nx = x;
    end
elseif a==3%Down
    ny = y - 1;
    if y==1
        ny = y;
    end
elseif a==4%Left
    nx = x - 1;
    if x==1
        nx = x;
    elseif x==2 && (y==4 || y==5)
        nx = x;
    elseif x==3 && y==5
        nx = x;
    end
end

if h == 1%H1
    if nx==3 && ny==4
        nd = 2;
    end
elseif h == 2%H2
    if nx==2 && ny==2
        nd = 2;
    end
elseif h == 3%H3
    if nx==4 && ny==2
        nd = 2;
    end
end

if nx==2 && ny==5
    nd = 1;%1 IN GOOD CONDITION(REPAIRED) 2 DAMAGED
end

end
