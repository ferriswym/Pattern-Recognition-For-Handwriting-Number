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
    %������ȡ
    [featuretemp,bmg,flag] = getFeature(X(i,:));        
    feature(i,:) = featuretemp;
end
%��һ��
[feature,se] = mapminmax(feature);                      
%����ѵ�����ݺͲ�������
trainX = feature(1:4000,:);
testX = feature(4001:5000,:);
trainY = y(1:4000,:);
testY = y(4001:5000,:);

%% neural network
tic;
spread = 0.1;
%������ѵ��������
net = newpnn(trainX',ind2vec(trainY'),spread);          
ti = toc;
fprintf('��������ģ�͹���ʱ %f sec\n', ti);

%% predict
lab0 = net(trainX');
lab = vec2ind(lab0);
rate = sum(lab' == trainY) / length(trainY);
fprintf('ѵ����������ȷ��Ϊ\n %d%%\n', round(rate*100));
lab0 = net(testX');
lab = vec2ind(lab0);
rate = sum(lab' == testY) / length(testY);
fprintf('������������ȷ��Ϊ\n %d%%\n', round(rate*100));