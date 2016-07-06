function [ Xnorm ] = normalise( X )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    
    Xnorm = zeros(size(X));
    
    for i = 2 : size(X,2)
        mean = sum(X(:,i))/size(X,1);
        Xnorm(:,i) = (X(:,i)- mean)/(max(X(:,i))-min(X(:,i)));
    end

end

