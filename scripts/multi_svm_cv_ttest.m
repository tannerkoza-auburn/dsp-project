function [hit_rate]=multi_svm_cv_ttest(group1,group2,group3,feature_number)

%%% SVM cross validation using features slected via t-test
%%% not all the features will be used, instead, only use 
%%% the highest "feature_number" (e.g. 100) features
%%% each cv fold may use different features as the training sets are
%%% slightly different in each fold
%%% the "com_feat_indx" returns the indeces of features that were commonly
%%% used in all the cv folds
%%% in "com_feat_indx", higher ranking feature is put on the left, the rank
%%% is determined by the mean rank across all the cv fold


all_features_X = [group1; group2; group3];
n_subs = size(all_features_X, 1);
all_features_Y = [];

for i=1:size(group1,1)
    all_features_Y = [all_features_Y; 0];
end
for i=1:size(group2,1)
    all_features_Y = [all_features_Y; 1];
end
for i=1:size(group3,1)
    all_features_Y = [all_features_Y; 2];
end

%labels = categorical(all_features_Y);
%encode_Y = onehotencode(labels, 2);

hit_rate=0;
rank=zeros(size(all_features_X,1),size(all_features_X,2)); 

for sbj = 1:1:n_subs
    test_X=all_features_X(sbj,:); test_Y=all_features_Y(sbj); % seperate the train and test data
    
    train_X=all_features_X; train_Y=all_features_Y; % train=all-test
    train_X(sbj,:)=[]; train_Y(sbj)=[];
    
    train_X1=train_X(train_Y == 1, :);
    train_X2=train_X(train_Y == 2, :);
    train_X3=train_X(train_Y == 3, :);    
    
    for i=1:size(train_X,2)
        p(i)=anova1([train_X1(:,i);train_X2(:,i);train_X3(:,i)],[ones(size(train_X1,1),1); 2*ones(size(train_X2,1),1); 3*ones(size(train_X3,1),1)],'off'); % group t-test on each feature
    end
    
    for r=1:1:feature_number       % features indices are listed in "rank" with higher rank on left 
        rank(sbj,r)=find(p(1,:)==min(p(1,:)));
        p(1,rank(sbj,r))=9999;
    end
    
    trn_X=[]; tst_X=[]; % prepare the training and testing sets with specified number of features
    for fe=1:1:feature_number
        trn_X=[trn_X, train_X(:,rank(sbj,fe))];
        tst_X=[tst_X, test_X(:,rank(sbj,fe))];    
    end
    

    t = templateSVM("Standardize", true, "KernelFunction", "rbf"); %Define machine learning model
    SVM = fitcecoc(trn_X, train_Y, "Learners", t, "Coding", "onevsall"); %Train the model
    predict_x = SVM.predict(tst_X);
    if predict_x == test_Y(1)
         hit_rate=hit_rate+1;
    end
    
    sbj
    feature_number
end

hit_rate=hit_rate/size(all_features_X,1);

end