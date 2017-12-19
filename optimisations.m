%inputs x and y
x = [-2:0.001:2];
y = [-2:0.001:2];

%inputs x and y in one matrix
X = [x' y'];

%true_output
true_out = exp(-x.^2 - y.^2)';

figure;
plot(true_out);

%number of layers
input  = 2;
hidden = 10;
true_output = 1;

%weights
weights1 = rand(hidden, input + 1);
weights2 = rand(true_output, hidden + 1);

weights = [weights1(:) ; weights2(:)];

%feed forward

for i = 1:500

weights1 = reshape(weights(1:hidden * (input + 1)), hidden, (input + 1));

weights2 = reshape(weights((1 + (hidden * (input + 1))):end), true_output, (hidden + 1));

m = size(X, 1);

cost = 0;
predictions1_grad = zeros(size(weights1));
predictions2_grad = zeros(size(weights2));

%each weight multiplied by input
act1 = [ones(m, 1) X];
val2 = act1 * weights1';
act2 = [ones(size(val2, 1), 1) sigmoid(val2)];
val3 = act2 * weights2';
prediction = sigmoid(val3);

sqrErr = (prediction - true_out).^2;

cost = 1 / (2*m) * sum(sqrErr); % the error function

%back propagation
error3 = prediction - true_out;
error2 = (error3 * weights2 .* sigmoidGradient([ones(size(val2, 1), 1) val2]));
error2=error2(:, 2:end);

grad1 = error2' * act1;
grad2 = error3' * act2;

predictions1_grad = grad1 ./ m;
predictions2_grad = grad2 ./ m ;

grad = [predictions1_grad(:) ; predictions2_grad(:)];

weights = weights - 3 .* grad;

end

m = size(X, 1);

act1 = [ones(m, 1) X];
val2 = act1 * weights1';
act2 = [ones(size(val2, 1), 1) sigmoid(val2)];
val3 = act2 * weights2';
prediction = sigmoid(val3);

figure2; plot(prediction);
