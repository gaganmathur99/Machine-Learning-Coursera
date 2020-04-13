function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h_theta=zeros(m);
sum1=zeros(m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h_theta=sigmoid(X*theta);
sum1=-y.*log(h_theta)-(1-y).*log(1-h_theta);
J=(1/m)*sum(sum1)+(lambda/(2*m))*sum(theta(2:size(theta)).^2);


grad=(1/m)*X'*(h_theta-y);
for i =2:size(grad)
  grad(i)=grad(i)+(lambda/m)*theta(i);
endfor


% =============================================================

end
