function [result_train_naive,result_test_naive,train_label_naive,test_label_naive] = Naive_Bayes_mnist(new_train,new_label,new_test,test_label,weight)
%importing data
%%%
%train_data=importdata('data_mnist_train.mat');
%test_data=importdata('data_mnist_test.mat');
%new_train = train_data.X_train;
%new_label = train_data.Y_train;
%new_test = test_data.X_test;
%test_label = test_data.Y_test;
%%%

%new_train(new_train>0) = 1;
%new_test(new_test>0) = 1;

%weight = rand(60000,1);

%weight = weight / sum(weight);
%weight(find(ismember(weight,mode(weight)))) = 1.67e-5;
weight = weight *1.3e4;
weight_train = exp(weight)/sum(exp(weight));

%weight/(max(weight)-min(weight));

%%
%training and testing

%calculate the prior probability of each class
ClassLabel=unique(new_label);
num_of_class = size(ClassLabel,1);
prior = zeros(num_of_class,1);
cls_train=cell(num_of_class,1);
cls_test = cell(num_of_class,1);
cls_all = zeros(num_of_class,1);
cls_weight_train=cell(num_of_class,1);
for i=1:num_of_class
    prior(i)=sum(ismember(new_label,ClassLabel(i)))/size(new_label,1);
    cls_weight_train{i}= weight_train(find(ismember(new_label,ClassLabel(i))),:);
    cls_train{i}=new_train(find(ismember(new_label,ClassLabel(i))),:)+1/60000;
    cls_test{i}=new_test(find(ismember(test_label,ClassLabel(i))),:)+1/60000;
end


%calculate the probability
for i = 1:size(new_test,2)
    new(i) = size(unique(new_test(:,i)),1);%how many different value for each feature in testing dataset
end

prob = cell(num_of_class,1);
for i=1:num_of_class
    prob{i}=zeros(size(new_test,2),max(new))+1/60000;
   % for j=1:size(new_test,2)     
   %  prob{i}(j,:)= 1/new(j); 
   % end
end

for p=1:num_of_class
for i = 1:size(cls_train{p},2)%for each feature
        value = unique(cls_train{p}(:,i)); %the list of value of one feature 
        for j = 1:size(unique(cls_train{p}(:,i)),1)%for each value within feature
            prob{p}(i,j) = prob{p}(i,j) + sum(cls_weight_train{p}.*ismember(cls_train{p}(:,i),value(j)))/sum(cls_weight_train{p});
        end
end
end

probability = cell(num_of_class,1);
for i =1:num_of_class
probability{i}=zeros(size(new_test,1),size(new_test,2));
end

%testing
for i=1:num_of_class
    for j=1:size(new_test,2)
        value = unique(new_test(:,j));
        %for p=1:size(new_test,1)
            %probability{i}(p,j) = prob{i}(j,find(value == new_test(p,j)));
            [~,b]=ismember(new_test(:,j),value);     
            probability{i}(:,j) = prob{i}(j,b);
        %end
    end
end


pred = zeros(size(new_test,1),size(prior,1));
for i =1:num_of_class
    %fin_prob = sum(log(probability{i}),2);%the last column is the times of all probability
    fin_prob = cumprod(probability{i},2);
    pred(:,i) = prior(i).*fin_prob(:,end);
end
[~,max_ind] = max(pred,[],2);
result = max_ind -1;
CCR = sum(result==test_label)/size(result,1);



%testing on training data
for i = 1:size(new_train,2)
    new_tr(i) = size(unique(new_train(:,i)),1);%how many different value for each feature in testing dataset
end

prob_train = cell(num_of_class,1);
for i=1:num_of_class
    prob_train{i}=zeros(size(new_train,2),max(new_tr))+1/60000;
end

%prob_train = prob;
for p=1:num_of_class

for i = 1:size(cls_train{p},2)%for each feature
    value_train = unique(cls_train{p}(:,i)); %the list of value of one feature 
    for j = 1:size(unique(cls_train{p}(:,i)),1)%for each value within feature
        prob_train{p}(i,j) =prob_train{p}(i,j) +   sum(cls_weight_train{p}.*ismember(cls_train{p}(:,i),value_train(j)))/sum(cls_weight_train{p});
        %prob_train{p}(i,j) =prob_train{p}(i,j) + sum(ismember(cls_train{p}(:,i),value_train(j)))/size(cls_train{p}(:,i),1);
        %prob_train{p}(i,j) = prob_train{p}(i,j) + sum(cls_weight_train{p}.*ismember(cls_train{p}(:,i),value_train(j)))/length(cls_train{p}(:,i));
    end
end

end


probability_train = cell(num_of_class,1);
for i =1:num_of_class
probability_train{i}=zeros(size(new_train,1),size(new_train,2));
end

%testing
for i=1:num_of_class

    for j=1:size(new_train,2)
        value_train = unique(new_train(:,j));
        %for p=1:size(new_train,1)
           % probability_train{i}(p,j) = prob_train{i}(j,find(value_train == new_train(p,j)));
        %end
         [~,b_train]=ismember(new_train(:,j),value_train);     
         probability_train{i}(:,j) = prob_train{i}(j,b_train);
    end

end

pred_train = zeros(size(new_train,1),size(prior,1));
for i =1:num_of_class
    %fin_prob_train = cumprod(probability_train{i},2);%the last column is the times of all probability
    fin_prob_train = sum(log(probability_train{i}),2);
    pred_train(:,i) = prior(i).*fin_prob_train(:,end);
end
[~,max_ind_train] = max(pred_train,[],2);
result_train = max_ind_train -1;
CCR_train = sum(result_train==new_label)/size(result_train,1);

train_label_naive = new_label;
test_label_naive = test_label;
result_train_naive=result_train;
result_test_naive=result;

end