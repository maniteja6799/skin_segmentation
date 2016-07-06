function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

m2= m*2;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


xtheta = sigmoid(X*theta);

for in=1:m
    J= J+ (-1*((y(in)*log(xtheta(in)))+((1-y(in))*(log(1-xtheta(in))))));
end
J=J/m;

temp=theta.^2;
jtemp=0;

%for i=2:size(theta)
jtemp=jtemp+sum(temp(2:end));
%end

jtemp = jtemp*(lambda/m2);
J = J+jtemp;

fprintf('Cost is %.5f\n',J);

hxy=xtheta-y;

grad(1)=0;
temp=hxy.*X(:,1);

%for j=1:size(temp)
    grad(1)=grad(1)+sum(temp);
%end

grad(1) = grad(1)/m;

for i = 2:size(theta)
    temp=hxy.*X(:,i);
    %for j=1:size(temp)
        grad(i)=grad(i)+sum(temp);
    %end
    grad(i)= grad(i)/m;
    grad(i)=grad(i)+((lambda*theta(i))/m);
end


end
