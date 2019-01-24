% Cost function
%
%   Cost function present in the Matlab Help for fminunc function

function [f,g] = costfunction(x)
% BROWNFG Nonlinear minimization test problem
%
% Evaluate the function
n = length(x); 
y = zeros(n,1);
i = 1:(n-1);
y(i)=(x(i).^2).^(x(i+1).^2+1) + (x(i+1).^2).^(x(i).^2+1);

f=sum(y);

% Evaluate the gradient if nargout > 1
if nargout > 1
    i=1:(n-1); 
    g = zeros(n,1);
    g(i) = 2*(x(i+1).^2+1).*x(i).* ...
        ((x(i).^2).^(x(i+1).^2))+ ...
        2*x(i).*((x(i+1).^2).^(x(i).^2+1)).* ...
        log(x(i+1).^2);
    g(i+1) = g(i+1) + ...
        2*x(i+1).*((x(i).^2).^(x(i+1).^2+1)).* ...
        log(x(i).^2) + ...
        2*(x(i).^2+1).*x(i+1).* ...
        ((x(i+1).^2).^(x(i).^2));
end
end