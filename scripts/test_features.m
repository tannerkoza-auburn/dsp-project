clear
clc
close all

project_dir = fileparts(which(mfilename)) + "/../";
addpath(genpath(project_dir))

load("features.mat")

[~, nfeatures] = size(data_class0_features);
hit_rate = multi_svm_cv_ttest(data_class0_features, data_class1_features, data_class2_features, nfeatures);
