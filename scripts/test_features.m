clear
clc
close all

project_dir = fileparts(which(mfilename)) + "/../";
addpath(genpath(project_dir))

load("features.mat")

[~, nfeatures] = size(data_class0);
[hit_rate, svm] = multi_svm_cv_ttest(data_class0, data_class1, data_class2, nfeatures);