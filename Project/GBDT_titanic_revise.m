%loading data
fid_train = fopen("new_titanic_train.csv");
Title_train = textscan(fid_train, '%s %s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
Data_train = textscan(fid_train, '%d %d %d %s %d %f %d %d %s %f %d','delimiter', ',');
fclose(fid_train);
fid_test = fopen("new_titanic_test.csv");
Title_test = textscan(fid_test, '%s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
Data_test = textscan(fid_test, '%d %d %s %d %f %d %d %s %f %d','delimiter', ',');
fclose(fid_test);
%%%
% for training data 1.'PassengerID' 2.'Survived' 3.'Pclass' 4.'Name' 5.'Sex' 6.'Age' 7.'SibSp' 8.'Parch' 9.'Ticket' 10.'Fare' 11.'Embarked'
%the passengerID,Name,Ticket is useless we drop these data
new_train = zeros(int16(size(Data_train{1},1)*0.8),3);
new_label = zeros(1,int16(size(Data_train{2},1)*0.8));
new_label = Data_train{2}(1:int16(size(Data_train{2},1)*0.8));
%for new_train :1.Pclass 2.Sex 3.Age 4.SibSp 5.Parch 6.Fare 7.Embarked
test_label = zeros(1,int16(size(Data_train{2},1)*0.2));
test_label = Data_train{2}(int16(size(Data_train{2},1)*0.8)+1:end);
%for new_train :1.Pclass 2.Sex 3.Age 4.SibSp 5.Parch 6.Fare 7.Embarked
new_train(:,1) =  Data_train{3}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,2) =  Data_train{5}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,3) =  Data_train{6}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,4) =  Data_train{7}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,5) =  Data_train{8}(1:int16(size(Data_train{2},1)*0.8));
new_train(:,6) =  Data_train{10}(1:int16(size(Data_train{2},1)*0.8));

new_train(:,7) =  Data_train{11}(1:int16(size(Data_train{2},1)*0.8));
new_test = zeros(int16(size(Data_train{1},1)*0.2),3);
new_test(:,1) = Data_train{3}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,2) = Data_train{5}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,3) = Data_train{6}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,4) =  Data_train{7}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,5) =  Data_train{8}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,6) =  Data_train{10}(int16(size(Data_train{2},1)*0.8)+1:end);
new_test(:,7) =  Data_train{11}(int16(size(Data_train{2},1)*0.8)+1:end);

n=size(new_train,1);%多少个训练样本
s=size(new_test,1);%number of test samples
ClassLabel=unique(new_label);
c = size(ClassLabel,1);%多少个类

new_label(new_label ==0 )= -1;
test_label(test_label ==0 )= -1;
%% Created by Indiffer
tree = fitctree(new_train,new_label,'MinLeafSize',20);
prediction_test = predict(tree,new_test);
ccr_test = sum(prediction_test == test_label)/length(test_label);
CCR_test(1)=ccr_test;
prediction_train = predict(tree,new_train);
ccr_train = sum(prediction_train == new_label)/length(new_label);
CCR_train(1)=ccr_train;


%F_k= rand(n,1);
%F_k = F_k/sum(F_k);
prediction_train(prediction_train == 0) = -1;
F_k= ones(n,1);
%%%%F_k_test=predict(tree,new_test);
%%%%F_k_test(F_k_test == 0) = -1;
y_hat = double(prediction_train);
y_hat = 2.*double(y_hat)./(1+exp(2.*double(y_hat).*F_k));
F_k = real(0.5.*log((1+y_hat)./(1-y_hat)));
%%
F_k_test= ones(s,1);
prediction_test=predict(tree,new_test);
prediction_test(prediction_test == 0) = -1;
y_hat_test = double(prediction_test);
y_hat_test = 2.*double(y_hat_test)./(1+exp(2.*double(y_hat_test).*F_k_test));
F_k_test = real(0.5.*log((1+y_hat_test)./(1-y_hat_test)));
%%
p_pos = 1./(1+exp(-2.*F_k));
p_neg = 1./(1+exp(2.*F_k));
a = 2*(p_pos>p_neg)-1;
CCR_= sum(a  == new_label)/length(new_label);
%F_k = real(0.5.*log(double(new_label)./double(1-new_label)));
m=50;
l = 1;
for i = 1:m
    for j = 1:c
        y_hat = 2.*double(new_label)./(1+exp(2.*double(new_label).*F_k));
        %%%%y_hat_test=2.*double(test_label)./(1+exp(2.*double(test_label).*F_k_test));
        rtree = fitctree(new_train,y_hat);
        [y,~,node] = predict(rtree,new_train);      
        [y_test,~,node_test] = predict(rtree,new_test);      
        node_unique = unique(node);
        t = ones(n,1);
        t_test=zeros(s,1);
        for k = 1:length(node_unique)
            y_hat_node = y_hat(ismember(node,node_unique(k)));
            y_node = double(new_label(ismember(node,node_unique(k))));
            yy(:,k) = ( sum(y_hat_node)/sum(abs(y_hat_node).*(2-abs(y_hat_node))) );
            t(ismember(node,node_unique(k))) = yy(k);
            t_test(ismember(node_test,node_unique(k)))=yy(k);
            
        end
        F_k = F_k + t;
        F_k_test=F_k_test+t_test;
        
    end
    p_pos = 1./(1+exp(-2.*F_k));
    p_neg = 1./(1+exp(2.*F_k));
    a = 2*(p_pos>p_neg)-1;
    CCR_train(l+1)= sum(a  == new_label)/length(new_label);
    %%
    p_pos_test = 1./(1+exp(-2.*F_k_test));
    p_neg_test = 1./(1+exp(2.*F_k_test));
    a_test = 2*(p_pos_test>p_neg_test)-1;
    CCR_test(l+1)= sum(a_test  == test_label)/length(test_label);
    %%
    l = l+1;
end

confusion_mat_train=confusionmat(new_label,int32(a));
recall_train=confusion_mat_train(2,2)/(confusion_mat_train(2,2)+confusion_mat_train(1,2));
precision_train=confusion_mat_train(2,2)/(confusion_mat_train(2,1)+confusion_mat_train(2,2));
F_score_train = 2*precision_train*recall_train/(precision_train+recall_train);
confusion_mat_test=confusionmat(test_label,int32(a_test));
recall_test=confusion_mat_test(2,2)/(confusion_mat_test(2,2)+confusion_mat_test(1,2));
precision_test=confusion_mat_test(2,2)/(confusion_mat_test(2,1)+confusion_mat_test(2,2));
F_score_test = 2*precision_test*recall_test/(precision_test+recall_test);




%figure(1);
%plot(1:(l),CCR_train,'b');
%hold on
%plot(1:(l),CCR_test,'r');
%legend('train CCR','test CCR')
%xlabel('iteration');
%ylabel('CCR');
%title('GBDT');


