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
ClassLabel=unique(new_label);
c = size(ClassLabel,1);%多少个类

%% Created by Indiffer

F_k= zeros(n,c);%c=2

m=3;
for i = 1:m
    for j = 1:c
        p(:,j) = exp( F_k(:,j) ) ./ sum ( exp( F_k ) ,2 );
    end
    for j = 1:c
        y_hat = double(new_label) - p(:,j);
        %y_hat_train(:,j) = double(test_label) - p(:,j);
        rtree = fitctree(new_train,y_hat,'MinLeafSize',20);
        [y_hat,~,node] = predict(rtree,new_train);
        %y_hat_train(:,j) = predict(rtree,new_train);
        node_unique = unique(node);
        for k = 1:length(node_unique)
            y_hat_node = y_hat(ismember(node,node_unique(k)));
            yy(:,k) = ((j-1)/j)*( sum(y_hat_node)/sum(abs(y_hat_node).*(1-abs(y_hat_node))) );
            %t = zeros(n,1);
            %t(ismember(node,node_unique(k))) = 1;
            %yy_sum = yy_sum + sum(yy).*t;
        end
        %yy = ((j-1)/j)*( sum(y_hat) / sum(abs(y_hat).*(1-abs(y_hat))) );
        F_k(:,j) = F_k(:,j) + sum(yy).*ones(n,1);
    end
end
pre_test = predict(rtree,new_test);
pre_train = predict(rtree,new_train);
CCR_test = sum(pre_test  == test_label)/length(test_label);
CCR_train = sum(pre_train  == new_label)/length(new_label);


