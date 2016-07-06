function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
m2= 2*m;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
xtheta = (X*theta);
for j=1:size(xtheta)
    xtheta(j,1) = sigmoid(xtheta(j,1));
end

for in=1:m
    J= J+ -1*((y(in)*log(xtheta(in)))+((1-y(in))*(log(1-xtheta(in)))));
end
J=J/m;
fprintf('Cost is %.2f\n',J);

hxy=xtheta-y;

for i=1:size(theta)
    temp=hxy.*X(:,i);
    for j=1:size(temp)
        grad(i)=grad(i)+temp(j);
    end
    grad(i)= grad(i)/m;
end


% =============================================================

end
