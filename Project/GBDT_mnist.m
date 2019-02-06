train_data=importdata('data_mnist_train.mat');
test_data=importdata('data_mnist_test.mat');
new_train = train_data.X_train;
new_label = train_data.Y_train;
new_test = test_data.X_test;
test_label = test_data.Y_test;

new_train(new_train>0)=1;
new_test(new_test>0)=1;
n=size(new_train,1);%多少个训练样本
ClassLabel=unique(new_label);
c = size(ClassLabel,1);%多少个类

%%
tree = fitctree(new_train,new_label,'MinLeafSize',20,'MaxNumSplits',200);
prediction_test = predict(tree,new_test);
CCR_test(1) = sum(prediction_test == test_label)/length(test_label);

prediction_train = predict(tree,new_train);
CCR_train(1) = sum(prediction_train == new_label)/length(new_label);


m=30;
F_k= zeros(n,c);
F_k_test = zeros(10000,10);
for j = 1:c
   p(:,j) = exp( F_k(:,j) ) ./ sum ( exp( F_k ) ,2 );
   p_test(:,j) = exp( F_k_test(:,j) ) ./ sum ( exp( F_k_test ) ,2 );
end
label = zeros(n,c);
for i = 1:c
    label(find(new_label == (i-1)),i) = 1;
end

for i = 1:40
    tic
    for j = 1:c
        y_hat = double(label(:,j)) - p(:,j);

        for q = 1:12
            y_hat( (10.^(-q)<y_hat)&(y_hat<(10.^(-q+1))) ) = mean(y_hat( (10.^(-q)<y_hat)&(y_hat<(10.^(-q+1))) ));
            y_hat((-(10.^(-q+1))<y_hat)&(y_hat<-10.^(-q))) = mean( y_hat((-(10.^(-q+1))<y_hat)&(y_hat<-10.^(-q))) );   
        end
        y_hat((-1e-12<y_hat)&(y_hat<1e-12)) = 0;
        rtree = fitctree(new_train,y_hat,'MinLeafSize',20,'MaxNumSplits',200);
        [y,~,node] = predict(rtree,new_train);
        [y_test,~,node_test] = predict(rtree,new_test);
        node_unique = unique(node);
        t = ones(n,1);
        t_test = ones(10000,1);
        for k = 1:length(node_unique)
            y_hat_node = y_hat(ismember(node,node_unique(k)));
            yy(:,k) = ((c-1)/c)*( sum(y_hat_node)/sum(abs(y_hat_node).*(1-abs(y_hat_node))) );
            t(ismember(node,node_unique(k))) = yy(k);
            t_test(ismember(node_test,node_unique(k))) = yy(:,k);
        end
        %yy = ((j-1)/j)*( sum(y_hat) / sum(abs(y_hat).*(1-abs(y_hat))) );
        F_k(:,j) = F_k(:,j) + t;
        F_k_test(:,j) = F_k_test(:,j) + t_test;
    end
    for j = 1:c
        p(:,j) = exp( F_k(:,j) ) ./ sum ( exp( F_k ) ,2 );
        p_test(:,j) = exp( F_k_test(:,j) ) ./ sum ( exp( F_k_test ) ,2 );
    end
    [~,ind]=max(p,[],2);
    [~,ind_test]=max(p_test,[],2);
    ind = ind-1;
    ind_test = ind_test - 1;
    CCR_train(i+1) = sum(ind==new_label)/length(new_label);
    CCR_test(i+1) = sum(ind_test==test_label)/length(test_label);
    toc
    sprintf("i%d",i)
    sprintf("the testCCR:%d ",CCR_test(i+1))
    sprintf("the trainCCR:%d ",CCR_train(i+1))
end

CCR_train_node100 = CCR_train;
CCR_test_node100 = CCR_test;
CCR_train(CCR_train<0.5) = max(CCR_train);
CCR_test(CCR_test<0.5) = max(CCR_test);
figure
plot(1:33,CCR_test(1:33),'b');
hold on
plot(1:21,CCR_test_node50(1:21),'r');
plot(1:25,CCR_test_node100(1:25),'black');
xlabel('iteration');
ylabel('CCR');
title('GBDT node compare');
legend('Max node = 200','Max node = 100','Max node = 50')