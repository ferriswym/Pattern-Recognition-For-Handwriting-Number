close all;
clear;
clc;
load('sample.mat');
% result = DecisionTree(trainX,trainY,testX,testY);   %������
% result = NeuroNetwork(trainX,trainY,testX,testY);   %������
% result = SVMDecision(trainX,trainY,testX,testY);    %SVM
% result = BayesNaive(trainX,trainY,testX,testY);     %Naive Bayes
% result = AdaBoost(trainX,trainY,testX,testY);       %AdaBoost
result = RandomForest(trainX,trainY,testX,testY);   %random forest