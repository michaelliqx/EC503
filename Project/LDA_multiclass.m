function[Y_LDA_train]=LDA_multiclass(D)
tic
numofClass=10;
%load('titanic_data.mat');
load('data_mnist_test');
load('data_mnist_train');
tic
% [coef_train score_train latent_train]=princomp(X_train);
% [coef_test score_test latent_test]=princomp(X_test);
% X_train=score_train(:,1:300);
% X_test=score_test(:,1:300);
X_train(X_train(:,:)>1)=1;
X_test(X_test(:,:)>1)=1;
%X_train=sparse(X_train);
%X_test=sparse(X_test);
toc
%D=randi(10,length(train),1);
size_data=size(X_train);
D=ones(size_data(1),1);
%D(505)=1000;
LDAmodel=LDA_train_mc(X_train,Y_train,numofClass,size_data(2),D);
[row_test,~]=size(X_test);
[row_train,~]=size(X_train);
P_test=zeros(numofClass,row_test);
P_train=zeros(numofClass,row_train);
for i =1:numofClass
    e = X_test-repmat(LDAmodel.Mu(i,:),row_test,1);
    e(e<1e-10) = 0;
    x_mean_test=sparse(e);
    P_test(i,:)=diag(1/2*x_mean_test*(inv(LDAmodel.Sigma))*x_mean_test')'-log(LDAmodel.Pi(i,1))*ones(1,row_test);
%     x_train=sparse(X_train-repmat(LDAmodel.Mu(i,:),row_train,1));
    f = X_train-repmat(LDAmodel.Mu(i,:),row_train,1);    
    f(f<1e-10) = 0;
    x_mean_train=sparse(f);
%     k=round(size_data(1)/5);
%      for j=1:5
         tic
        %x_mean_train=sparse(x_train(k*(j-1)+1:k*j,:));
        %x_mean_train=x_train;
        %P_train(i,k*(j-1)+1:k*j)=diag(1/2*x_mean_train*(inv(LDAmodel.Sigma))*x_mean_train')'-log(LDAmodel.Pi(i,1))*ones(1,k);
        a=sparse((inv(LDAmodel.Sigma))*x_mean_train');
        P_train(i,:)=diag(1/2*x_mean_train*a)'-log(LDAmodel.Pi(i,1))*ones(1,k);
        toc
%      end
end
[~,Y_LDA_test]=min(P_test);
[~,Y_LDA_train]=min(P_train);

% Y_LDA_train=power(-1,Y_hat_train);
CM_test=confusionmat(Y_LDA_test-1,Y_test);
CCR_test=sum(diag(CM_test))/length(Y_test)
% precision_test=CM_test(2,2)/(CM_test(2,2)+CM_test(2,1))
% recall_test=CM_test(2,2)/(CM_test(2,2)+CM_test(1,2))
% F_score_test=2*precision_test*recall_test/(precision_test+recall_test)

CM_train=confusionmat(Y_LDA_train-1,Y_train);
CCR_train=sum(diag(CM_train))/length(Y_train)
% precision_train=CM_train(2,2)/(CM_train(2,2)+CM_train(2,1))
% recall_train=CM_train(2,2)/(CM_train(2,2)+CM_train(1,2))
% F_score_train=2*precision_train*recall_train/(precision_train+recall_train)
toc
return
end