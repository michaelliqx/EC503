function [LDAmodel]= LDA_train_mc(X_train, Y_train, numofClass,features,D)
%
% Training QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes 
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:
%load('data_iris.mat')

train_num=length(X_train);
DAmodel.Mu=zeros(numofClass,features);
LDAmodel.Sigma=zeros(features,features);
LDAmodel.Pi=zeros(numofClass,1);
sum_D=sum(D);
D=D/sum_D;
%X_train=X_train;
%Y_train=Y_train.*D;
%calculate Mu, Sigma and Pi
for i =1:numofClass
    n=length(find(Y_train==i-1));%(-1)^i when binary
    X=X_train(find(Y_train==i-1),:)';
    d=D(find(Y_train==i-1));
    In=ones(n,1);
    %LDAmodel.Mu(i,:)=X*In/n;
    %LDAmodel.Mu(i,:)=(X.*d')*In;
    LDAmodel.Mu(i,:)=(X.*repmat(d',features,1))*In;
    %X_mean=X*(eye(n,n)-In*In'/n);
    X_mean=X-repmat(LDAmodel.Mu(i,:)',1,n);
    %X_mean=X_mean.*repmat(d',features,1);
    %LDAmodel.Sigma=X_mean*X_mean'/length(Y_train);
    sigma=X_mean.*repmat(d',features,1)*X_mean';
    LDAmodel.Sigma=LDAmodel.Sigma+sigma/numofClass;
    LDAmodel.Pi(i,1)=n/length(Y_train);
end
%LDAmodel.Sigma=0.5*LDAmodel.Sigma+0.5*eye(features,features);
return 