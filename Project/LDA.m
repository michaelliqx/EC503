function[Y_hat]=LDA()
numofClass=2;
M=csvread('D:\BU classes\EC503\project\new_titanic_train.csv',1,0);
train_num=791;
train=M(1:train_num,3:end);
test=M((train_num+1):end,3:end);
label_train=(M(1:train_num,2)-0.5)*2;
label_test=(M((train_num+1):end,2)-0.5)*2;

LDAmodel=LDA_train(train,label_train,2);
[row,~]=size(test);
P=zeros(numofClass,row);

for i =1:numofClass
    %disp(size(X_test))
    %disp(size(LDAmodel.Mu(i,:)))
    x_mean=test-repmat(LDAmodel.Mu(i,:),row,1);
    P(i,:)=diag(1/2*x_mean*(inv(LDAmodel.Sigma))*x_mean')'-log(LDAmodel.Pi(i,1))*ones(1,row);
end
[~,Y_hat]=min(P);
CM_test=confusionmat(label_test,power(-1,Y_hat))
CCR=sum(diag(CM_test))/length(label_test)
return
end