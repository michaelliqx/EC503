%function [result] = KNN(train,test,k)
%end

fid_train = fopen("new_titanic_train.csv");
Title_train = textscan(fid_train, '%s %s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
Data_train = textscan(fid_train, '%d %d %d %s %d %f %d %d %s %f %d','delimiter', ',');
fclose(fid_train);
fid_test = fopen("new_titanic_test.csv");
Title_test = textscan(fid_test, '%s %s %s %s %s %s %s %s %s %s',1,'delimiter', ',');
Data_test = textscan(fid_test, '%d %d %s %d %f %d %d %s %f %d','delimiter', ',');
fclose(fid_test);
% for training data 1.'PassengerID' 2.'Survived' 3.'Pclass' 4.'Name' 5.'Sex' 6.'Age' 7.'SibSp' 8.'Parch' 9.'Ticket' 10.'Fare' 11.'Embarked'
%the passengerID,Name,Ticket is useless we drop these data
new_train = zeros(size(Data_train{1},1),7);
new_label = zeros(1,size(Data_train{2},1));
new_label = Data_train{2};
%for new_train :1.Pclass 2.Sex 3.Age 4.SibSp 5.Parch 6.Fare 7.Embarked
new_train(:,1) =  Data_train{3};
new_train(:,2) =  Data_train{5};
new_train(:,3) =  Data_train{6};
new_train(:,4) =  Data_train{7};
new_train(:,5) =  Data_train{8};
new_train(:,6) =  Data_train{10};
new_train(:,7) =  Data_train{11};
%for testing data:1.'Pclass'2.'Sex' 3.'Age' 4.'SibSp' 5.'Parch'6.'Fare' 7.'Embarked'
new_test = zeros(size(Data_test{1},1),7);
new_test(:,1) = Data_test{2};
new_test(:,2) = Data_test{4};
new_test(:,3) = Data_test{5};
new_test(:,4) = Data_test{6};
new_test(:,5) = Data_test{7};
new_test(:,6) = Data_test{9};
new_test(:,7) = Data_test{10};

%%
n=size(new_train,1);
ClassLabel=unique(new_label);
k=5;
pi=zeros(2,size(new_test,1));
dist=zeros(n,1);
for j=1:size(new_test,1)
    cnt=zeros(2,1);
    for i=1:n
        dist(i)=norm(sparse(new_train(i,:))-sparse(new_test(j,:)));
     end
 
    [d,index]=sort(dist);
    for i=1:k
        ind=find(ClassLabel==new_label(index(i)));
        cnt(ind)=cnt(ind)+1;
    end
    pi(:,j)=cnt(:,1)/10;
    [m,ind]=max(cnt);
    target(j)=ClassLabel(ind);
    knnreturn=struct('pi',pi ,'target',target);
end

columns = {'PassengerID','survival'};
data = table(Data_test{1},target','VariableNames', columns);
writetable(data, 'submission.csv')
