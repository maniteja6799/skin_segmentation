function out = mapFeature(X1, X2 , X3)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

maxdegree = 2;
out = [ones(size(X1(:,1))) X1(:,1) X2(:,1) X3(:,1)];

for degree = 2:maxdegree
    for  i = 0:degree
        for j = 0:degree-i
            k = degree -i-j;
            out(:, end+1) = (X1.^(i)).*(X2.^j).*(X3.^k);
        end
    end
end


end