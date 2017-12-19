function g = sigmoidGradient(z)
% returns the gradient of the sigmoid function evaluated at z

g = zeros(size(z));

g = sigmoid(z).*(1 - sigmoid(z));

end
