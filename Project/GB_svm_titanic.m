function[]=SVM_GB()
warning off;
load('titanic_data.mat');
n=size(train,1);%training sample
s=size(test,1);%number of test samples


label_train(label_train ==0 )= -1;
label_test(label_test ==0 )= -1;
%% Created by Indiffer
[Y_svm_train,Y_svm_test]=multiclass_SVM(train, label_train,train,test);
%[Y_svm_train,Y_svm_test]=multisvm(train, label_train,train,test);
CCR_test(1)=sum(Y_svm_test == label_test)/length(label_test);
CCR_train(1)=sum(Y_svm_train == label_train)/length(label_train);

% tree = fitctree(train,label_train,'MinLeafSize',20);
% prediction_test = predict(tree,test);
% ccr_test = sum(prediction_test == label_test)/length(label_test);
% CCR_test(1)=ccr_test;
% prediction_train = predict(tree,train);
% ccr_train = sum(prediction_train == label_train)/length(label_train);
% CCR_train(1)=ccr_train;

F_k= ones(n,1);
y_hat = Y_svm_train;
y_hat = 2.*double(y_hat)./(1+exp(2.*double(y_hat).*F_k));
F_k = real(0.5.*log((1+y_hat)./(1-y_hat)));
%%
F_k_test= ones(s,1);
y_hat_test = Y_svm_test;
y_hat_test = 2.*double(y_hat_test)./(1+exp(2.*double(y_hat_test).*F_k_test));
F_k_test = real(0.5.*log((1+y_hat_test)./(1-y_hat_test)));
%%
% p_pos = 1./(1+exp(-2.*F_k));
% p_neg = 1./(1+exp(2.*F_k));
% a = 2*(p_pos>p_neg)-1;
% CCR_= sum(a  == label_train)/length(label_train);
%F_k = real(0.5.*log(double(label_train)./double(1-label_train)));
m=10;
l = 1;
for i = 1:m
    i
%gradient of loss function
    y_hat = 2.*double(label_train)./(1+exp(2.*double(label_train).*F_k));
    %train and testing
 %   [Y_svm_train,Y_svm_test]=multiclass_SVM(train, y_hat,train,test);
    [Y_svm_train,Y_svm_test]=multisvm(train, y_hat,train,test);
%         rtree = fitctree(train,y_hat);
%         [y,~,node] = predict(rtree,train);      
%         [y_test,~,node_test] = predict(rtree,test);  

     node_unique = unique(y_hat);
     t = ones(n,1);
     t_test=zeros(s,1);
     for k = 1:length(node_unique)
         %leaf samples
         y_hat_node = y_hat(ismember(Y_svm_train,node_unique(k)));
         %average of leaf samples
         yy(k) = ( sum(y_hat_node)/sum(abs(y_hat_node).*(2-abs(y_hat_node))) );
         %assign average of leaf samples to every test sample in train and
         %test
         t(ismember(Y_svm_train,node_unique(k))) = yy(k);
         t_test(ismember(Y_svm_test,node_unique(k)))=yy(k);
     end

    F_k = F_k + t;
    F_k_test=F_k_test+t_test;
        
    p_pos = 1./(1+exp(-2.*F_k));
    p_neg = 1./(1+exp(2.*F_k));
    a = 2*(p_pos>p_neg)-1;
    CCR_train(l+1)= sum(a  == label_train)/length(label_train);
    %%
    p_pos_test = 1./(1+exp(-2.*F_k_test));
    p_neg_test = 1./(1+exp(2.*F_k_test));
    a_test = 2*(p_pos_test>p_neg_test)-1;
    CCR_test(l+1)= sum(a_test  == label_test)/length(label_test);
    %%
    l = l+1;
end

figure(1);
plot(1:(l),CCR_train,'b');
hold on
plot(1:(l),CCR_test,'r');
legend('train CCR','test CCR')
xlabel('iteration');
ylabel('CCR');
title('Gradient Boost SVM on titanic');
end