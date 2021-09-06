function [ individual_out ] = mutate( individual_in )
%MUTATE Summary of this function goes here
%   Detailed explanation goes here
% individual_out = individual_in;
index = individual_in.depthfirstiterator;
node = index(randi(length(index)));
parent = individual_in.getparent(node);
subtree_out = individual_in.subtree(node);
max_depth = depth(subtree_out);
temp = individual_in.chop(node);
%% subtree
function_set = {'AND','OR','NOT'};
terminal_set = {'A','B'};
general_set = union(function_set,terminal_set);

mysubtree = tree(function_set(randi(3)));

    while mysubtree.depth<max_depth
        if rand(1)<.8
            index = mysubtree.depthfirstiterator;
            for j = index
                if mysubtree.isleaf(j)
                    flag = false;
                    hight = 1;
                    p = j;
                    while mysubtree.getparent(p)~=0
                        p = mysubtree.getparent(p);
                        hight = hight + 1;
                    end
                    if hight>=max_depth
                        flag = true;
                    end
                    if strcmp(mysubtree.get(j),'NOT')
                        if ~flag
                            mysubtree = mysubtree.addnode(j,general_set(randi(5)));
                        else
                            mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
                        end
                    elseif strcmp(mysubtree.get(j),'OR')||strcmp(mysubtree.get(j),'AND')
                        if ~flag
                            mysubtree = mysubtree.addnode(j,general_set(randi(5)));
                            mysubtree = mysubtree.addnode(j,general_set(randi(5)));
                        else
                            mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
                            mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
                        end
                    end
                end
            end
        else
            break;
        end
    end
    index = mysubtree.depthfirstiterator;
    for j = index
        if mysubtree.isleaf(j)
            if strcmp(mysubtree.get(j),'NOT')
                mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
            elseif strcmp(mysubtree.get(j),'OR')||strcmp(mysubtree.get(j),'AND')
                
                mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
                mysubtree = mysubtree.addnode(j,terminal_set(randi(2)));
                
            end
        end
    end
%% graft
    individual_out = temp.graft(parent, mysubtree);
end

