clear all;
load('data_mnist_train.mat')
load('data_mnist_test.mat')
X_train(X_train>0) = 1;
X_test(X_test>0) = 1;

[train_num, feature_dim] = size(X_train);
D = ones(train_num, 1);
D = D / sum(D);
tic
[Y_pred,Y_hat,~,~] = naive_bayes_mnist(X_train,Y_train,X_test,Y_test,D);
toc
cmat = confusionmat(Y_hat, Y_test);
ccr = sum(Y_hat==Y_test) / length(Y_test);
disp(ccr)


% hyperparameter
[train_num, feature_dim] = size(X_train);
iteration = 100;
numofclass = length(unique(Y_train));
% Initialization
D = ones(train_num, 1);
D = D / sum(D);
% weight for each iteration
weight_all = zeros(1, iteration);
prediction = zeros(length(Y_test), iteration);
for i = 1:iteration
    % Fit T
    tic
    [Y_pred,Y_hat,~,~] = naive_bayes_mnist(X_train,Y_train,X_test,Y_test,D);
    toc
    % calculate loss
    epsilon = (1 - sum(Y_train == Y_pred) / length(Y_train));
    % weight
    weight = log((1 - epsilon)/epsilon) + log(numofclass - 1);
    % renew distribution D
    d_new = D .* exp(weight * double(Y_train ~= Y_pred));
    D = d_new / sum(d_new);
    % save weight
    prediction(:, i) = Y_hat;
    weight_all(:, i) = weight;
    if mode(i, 10) == 0
        disp(i)
    end
end

wpre = zeros(numofclass, length(Y_test));
for j = 1:numofclass
    wpre(j, :) = sum(weight_all * double(prediction==j-1).', 1);
end
[~, pred] = max(wpre, [], 1);
pred = pred - 1;
ccr3 = sum(pred.'==Y_test) / length(Y_test);
disp(ccr3)