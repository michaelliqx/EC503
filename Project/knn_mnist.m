function [target] = KNN(X_train,Y_train,X_test1,k)
%loading data
%%%
train_data=importdata('data_mnist_train.mat');
test_data=importdata('data_mnist_test.mat');
X_train=train_data.X_train;
Y_train=train_data.Y_train;
X_test1=test_data.X_test;
Y_test=test_data.Y_test;
%%%
%%
distance=cell(10,1);
j=1;

%dividing the data into matrices of 1000*60000 using cells and computing
%distances
for i=0:9
X_test=X_test1((1+1000*i:1000+1000*i),:);
p=X_train.^2;
p=sum(p,2);
q=X_test.^2;
q=sum(q,2);
b=q*ones(1,60000);
a=ones(1000,1)*p';
distance{j,1}=b+a-(2.*(X_test*X_train'));
j=j+1;
end

labels=unique(Y_train);
count=zeros(10,1);
j=1;
for i=0:length(labels)-1
   count(j,1)=length(Y_train(Y_train(:,1)==i));
   j=j+1;
end
%finding indexes of labels
indexes=cell(10,1);
for i=1:length(distance)
   dist=distance{i,1};
   [sorted_dist ind]=min(dist,[],2);
   indexes{i,1}=ind;
end
final_labels=cell(10,1);

%vertically concatenating all the indexes of labels
b=vertcat(indexes{(1:10),1});
%finding labels
 for i=1:length(b)
     label=b(i);
     target(i,1)=Y_train(label,1);
 end 
 
CCR = sum(Y_test==target)/10000;
end