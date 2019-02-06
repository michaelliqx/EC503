function [LDAmodel]= LDA_train(X_train, Y_train, numofClass)
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


DAmodel.Mu=zeros(numofClass,7);
LDAmodel.Sigma=zeros(7,7);
LDAmodel.Pi=zeros(numofClass,1);


%calculate Mu, Sigma and Pi
for i =1:numofClass
    n=length(find(Y_train==(-1)^i));
    X=X_train(find(Y_train==(-1)^i),:)';
    In=ones(n,1);
    LDAmodel.Mu(i,:)=X*In/n;
    X_mean=X*(eye(n,n)-In*In'/n);
    LDAmodel.Sigma=X_mean*X_mean'/length(Y_train);
    LDAmodel.Pi(i,1)=n/length(Y_train);
end
return 