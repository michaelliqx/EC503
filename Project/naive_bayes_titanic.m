function [result_train_naive,result_test_naive,train_label_naive,test_label_naive] = Naive_Bayes(Data_train,Data_test,weight)
%loading data
%在主程序中调用下面的语句，然后将Data_train,Data_test,weight作为输入即可，会自动生成naive_bayes.mat
%这个文件，同时也会返回上面的结果
%%%
%fid_train = fopen("new_titanic_train.csv");
%Title_train = textscan(fid_train, '%s %s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
%Data_train = textscan(fid_train, '%d %d %d %s %d %f %d %d %s %f %d','delimiter', ',');
%fclose(fid_train);
%fid_test = fopen("new_titanic_test.csv");
%Title_test = textscan(fid_test, '%s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
%Data_test = textscan(fid_test, '%d %d %s %d %f %d %d %s %f %d','delimiter', ',');
%fclose(fid_test);
%%%

% for training data 1.'PassengerID' 2.'Survived' 3.'Pclass' 4.'Name' 5.'Sex' 6.'Age' 7.'SibSp' 8.'Parch' 9.'Ticket' 10.'Fare' 11.'Embarked'
%the passengerID,Name,Ticket is useless we drop these data
new_train = zeros(int16(size(Data_train{1},1)*0.8),4);
new_label = zeros(1,int16(size(Data_train{2},1)*0.8));
%new_label2 = zeros(178,1);
new_label = Data_train{2}(1:int16(size(Data_train{2},1)*0.8));
%new_label2 = Data_train{2}(int16(size(Data_train{2},1)*0.8)+1:end);
test_label = zeros(1,int16(size(Data_train{2},1)*0.2));
test_label = Data_train{2}(int16(size(Data_train{2},1)*0.8)+1:end);
%for new_train :1.Pclass 2.Sex 3.Age 4.SibSp 5.Parch 6.Fare 7.Embarked
new_train(:,1) =  Data_train{3}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,2) =  Data_train{5}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,3) =  Data_train{6}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,4) =  Data_train{7}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,5) =  Data_train{8}(1:int16(size(Data_train{2},1)*0.8));
%new_train(:,6) =  Data_train{10};
%new_train(:,7) =  Data_train{11};

%for testing data:1.'Pclass'2.'Sex' 3.'Age' 4.'SibSp' 5.'Parch'6.'Fare' 7.'Embarked'
new_test = zeros(int16(size(Data_train{1},1)*0.2),4);
%new_test(:,1) = Data_test{2};
%new_test(:,2) = Data_test{4};
%new_test(:,3) = Data_test{5};
%new_test(:,4) = Data_test{6};

new_test(:,1) = Data_train{3}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,2) = Data_train{5}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,3) = Data_train{6}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,4) = Data_train{7}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,5) = Data_train{8}(int16(size(Data_train{2},1)*0.8)+1:end);
%new_test(:,6) = Data_test{9};
%new_test(:,7) = Data_test{10};

%for these  two features we need to modify them
new_train(find(new_train(:,3)<20),3)=0;
new_train(find(20<=new_train(:,3)&new_train(:,3)<40),3)=1;
new_train(find(40<=new_train(:,3)&new_train(:,3)<60),3)=2;
new_train(find(60<=new_train(:,3)),3)=4;
new_test(find(new_test(:,3)<20),3)=0;
new_test(find(20<=new_test(:,3)&new_test(:,3)<40),3)=1;
new_test(find(40<=new_test(:,3)&new_test(:,3)<60),3)=2;
new_test(find(60<=new_test(:,3)),3)=4;

%new_train(find(new_train(:,6)<30),6)=0;
%new_train(find(new_train(:,6)<60&new_train(:,6)>=30),6)=1;
%new_train(find(new_train(:,6)<90&new_train(:,6)>=60),6)=2;
%new_train(find(new_train(:,6)<120&new_train(:,6)>=90),6)=3;
%new_train(find(new_train(:,6)>=120),6)=4;
%new_test(find(new_test(:,6)<30),6 )=0;
%new_test(find(new_test(:,6)<60&new_test(:,6)>=30),6)=1;
%new_test(find(new_test(:,6)<90&new_test(:,6)>=60),6)=2;
%new_test(find(new_test(:,6)<120&new_test(:,6)>=90),6)=3;
%new_test(find(new_test(:,6)>=120),6)=4;
%weight = rand(713,1);
weight =ones(713,1);
weight = weight/sum(weight);
weight_train = weight;
%weight_test = weight(int16(end*0.8)+1:end);
%%
%training

%calculate the prior probability of each class
ClassLabel=unique(new_label);%[0,1]
num_of_class = size(ClassLabel,1);
prior = zeros(num_of_class,1);
cls_train=cell(num_of_class,1);
cls_test = cell(num_of_class,1);
cls_all = zeros(num_of_class,1);
cls_weight_train=cell(num_of_class,1);
for i=1:num_of_class
    prior(i)=sum(ismember(new_label,ClassLabel(i)))/size(new_label,1);
    cls_train{i}=new_train(find(ismember(new_label,ClassLabel(i))),:);
    cls_weight_train{i}= weight_train(find(ismember(new_label,ClassLabel(i))),:);
    cls_test{i}=new_test(find(ismember(test_label,ClassLabel(i))),:);
    cls_all(i) = sum(ismember(Data_train{2},ClassLabel(i)));
end
%plot
%%%
%x = [1,2,3];
%y1 = [cls_all(1),length(cls_train{1}),length(cls_test{1})];
%y2 = [cls_all(2),length(cls_train{2}),length(cls_test{2})];
%bar(x,[y1;y2]')
%title('Dataset split')
%xlabel('Dataset')
%ylabel('Number')
%legend('Not Survived','Survived')
%set(gca,'xticklabel',{'original train','train','validation'});

%data=[length(new_label),length(test_label)];%输入数据
%label={'train','validation'};%输入标签
%explode=[1 0];%定义突出的部分
%p=data/sum(data);%计算比例
%split=num2str(p'*100,'%1.2f');%计算百分比
%split=[repmat(blanks(2),length(data),1),split,repmat('%',length(data),1)];
%split=cellstr(split);
%Label=strcat(label,split');
%title("Dataset Split");
%pie(data,explode,Label)
%%%

%calculate the probability
for i = 1:size(new_test,2)
    new(i) = size(unique(new_test(:,i)),1);%how many different value for each feature in testing dataset
end
prob = cell(num_of_class,1);
for i=1:num_of_class
    prob{i}=zeros(size(new_test,2),max(new))+1/max(new);
   % for j=1:size(new_test,2)
   % prob{i}(j,:)= 1/new(j); 
   % end
end
for p=1:num_of_class
    for i = 1:size(cls_train{p},2)%for each feature
        value = unique(cls_train{p}(:,i)); %the list of value of one feature 
        for j = 1:size(unique(cls_train{p}(:,i)),1)-1%for each value within feature
            prob{p}(i,j) = prob{p}(i,j) + sum(cls_weight_train{p}.*ismember(cls_train{p}(:,i),value(j)))/sum(cls_weight_train{p}.*cls_train{p}(:,i));
        end
    end
end

%testing
probability = cell(num_of_class,1);
for i =1:num_of_class
probability{i}=zeros(size(new_test,1),size(new_test,2));
end
for i =1: num_of_class%for each class
    for j=1:size(new_test,2)%for each feature
        value = unique(new_test(:,j));
        %for p = 1:size(new_test,1)%for each sample 
        %    probability{i}(p,j) = prob{i}(j,find(value == new_test(p,j)));
        %end
        [~,b]=ismember(new_test(:,j),value);     
        probability{i}(:,j) = prob{i}(j,b);
    end
end
pred = zeros(size(new_test,1),size(prior,1));
for i =1:num_of_class
    fin_prob = cumprod(probability{i},2);
    pred(:,i) = prior(i)*fin_prob(:,end);
end
[~,max_ind] = max(pred,[],2);
result = max_ind - 1;
result(find(result==0))=-1;

test_label(find(test_label==0))=-1;
CCR=sum(test_label==result)/size(result,1);



%testing on training data
for i = 1:size(new_train,2)
    new(i) = size(unique(new_train(:,i)),1);%how many different value for each feature in testing dataset
end
prob = cell(num_of_class,1);
for i=1:num_of_class
    prob{i}=zeros(size(new_train,2),max(new))+1/max(new);
   % for j=1:size(new_test,2)
   % prob{i}(j,:)= 1/new(j); 
   % end
end
for p=1:num_of_class
for i = 1:size(cls_train{p},2)%for each feature
    value = unique(cls_train{p}(:,i)); %the list of value of one feature 
    for j = 1:size(unique(cls_train{p}(:,i)),1)-1%for each value within feature
        prob{p}(i,j) =prob{p}(i,j) + sum(cls_weight_train{p}.*ismember(cls_train{p}(:,i),value(j)))/sum(cls_weight_train{p}.*cls_train{p}(:,i));
    end
end
end
probability_train = cell(num_of_class,1);
for i =1:num_of_class
probability_train{i}=zeros(size(new_train,1),size(new_train,2));
end
for i =1: num_of_class%for each class
    for j=1:size(new_train,2)%for each example
        value = unique(new_train(:,j));
        %for p = 1:size(new_train,1)%for each feature
        %    probability_train{i}(p,j) = prob{i}(j,find(value == new_train(p,j)));
        %end
        [~,b_train]=ismember(new_train(:,j),value);     
        probability_train{i}(:,j) = prob{i}(j,b_train);
    end
end
pred_train = zeros(size(new_train,1),size(prior,1));
for i =1:num_of_class
    fin_prob_train = cumprod(probability_train{i},2);
    pred_train(:,i) = prior(i).*fin_prob_train(:,end);
end
[~,max_ind_train] = max(pred_train,[],2);
result_train = max_ind_train - 1;
result_train(find(result_train==0))=-1;
new_label(find(new_label==0))=-1;
train_label_naive=new_label;
test_label_naive = test_label;
result_train_naive=result_train;
result_test_naive=result;
CCR_train=sum(new_label==result_train)/size(result_train,1);
%confusion matrix
confusion_mat_train=confusionmat(train_label_naive,int32(result_train_naive));
recall_train=confusion_mat_train(2,2)/(confusion_mat_train(2,2)+confusion_mat_train(1,2));
precision_train=confusion_mat_train(2,2)/(confusion_mat_train(2,1)+confusion_mat_train(2,2));
F_score_train = 2*precision_train*recall_train/(precision_train+recall_train);
confusion_mat_test=confusionmat(test_label_naive,int32(result_test_naive));
recall_test=confusion_mat_test(2,2)/(confusion_mat_test(2,2)+confusion_mat_test(1,2));
precision_test=confusion_mat_test(2,2)/(confusion_mat_test(2,1)+confusion_mat_test(2,2));
F_score_test = 2*precision_test*recall_test/(precision_test+recall_test);

save("naive_bayes.mat",'result_train_naive','result_test_naive',"train_label_naive","test_label_naive");
%columns = {'PassengerID','Survived'};
%data = table(Data_test{1},result,'VariableNames', columns);
%writetable(data, 'submission.csv')
end