%% preprocess
clear;
clc;
close all;
load('handwrite.mat');
thresh = graythresh(X);
X = X > thresh;
sel = randperm(size(X, 1));
X = X(sel,:);
y = y(sel,:);

%% displaydata
sel = sel(1:100);
displayData(X(sel, :));

%% feature extraction
feature = zeros(size(X,1),14);
for i = 1:size(X,1)
    %特征提取
    [featuretemp,bmg,flag] = getFeature(X(i,:));        
    feature(i,:) = featuretemp;
end
%归一化
[feature,se] = mapminmax(feature);                      
%分离训练数据和测试数据
trainX = feature(1:4000,:);
testX = feature(4001:5000,:);
trainY = y(1:4000,:);
testY = y(4001:5000,:);

%% neural network
tic;
spread = 0.1;
%建立并训练神经网络
net = newpnn(trainX',ind2vec(trainY'),spread);          
ti = toc;
fprintf('建立网络模型共耗时 %f sec\n', ti);

%% predict
lab0 = net(trainX');
lab = vec2ind(lab0);
rate = sum(lab' == trainY) / length(trainY);
fprintf('训练样本的正确率为\n %d%%\n', round(rate*100));
lab0 = net(testX');
lab = vec2ind(lab0);
rate = sum(lab' == testY) / length(testY);
fprintf('测试样本的正确率为\n %d%%\n', round(rate*100));