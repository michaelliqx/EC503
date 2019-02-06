x = rand(100, 7);
y = int32((sum(x, 2) > 3.5));
y(y==0) = -1;
x_test = rand(50, 7);
y_test = int32((sum(x_test, 2) > 3.5));
y_test(y_test==0) = -1;



NBModel = fitNaiveBayes(x,y);
y_pred_1 = predict(NBModel, x_test);
rda = fitcdiscr(x,y);
y_pred_2 = predict(rda, x_test);
ccr1 = sum(y_pred_1 == y_test) / length(y_pred_1);
ccr2 = sum(y_pred_2 == y_test) / length(y_pred_1);
disp(ccr1)
disp(ccr2)

% hyperparameter
[train_num, feature_dim] = size(x);
iteration = 10;
numofclassifier = 2;
% Initialization
D = ones(1, train_num) / train_num;
y_pred = [predict(rda, x) predict(NBModel, x)].';
y_rep = repmat(y.', numofclassifier, 1);
% weight for each iteration
weight_all = zeros(numofclassifier, iteration);
for i = 1:iteration
    % calculate loss
    epsilon = 1 - sum(repmat(D, numofclassifier, 1) .* (y_pred==y_rep), 2) ./ sum(repmat(D, numofclassifier, 1), 2);
    % get results of weak learner
    [prob, best_idx] = min(epsilon);
    % weight
    weight = zeros(numofclassifier, 1);
    weight(best_idx) = 0.5 * log(1 ./ epsilon(best_idx) - 1);
    % renew distribution D
    d_new = D .* exp(- weight.' * double(y_rep .* y_pred));
    D = d_new / sum(d_new);
    % save weight
    weight_all(:, i) = weight;
end

h_t = [predict(rda, x_test) predict(NBModel, x_test)].';
prediction = sign(sum(weight_all.' * double(h_t), 1));
ccr3 = sum(prediction==y_test.') / length(y_test);
disp(ccr3)

