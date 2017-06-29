%% preprocess
clear;
clc;
close all;
load('handwrite.mat');
thresh = graythresh(X);
X = X > thresh;

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
trainX = [];
testX = [];
trainY = [];
testY = [];
for i = 1:10
    trainX = [trainX;feature((i - 1) * 500 + 1:(i - 1) * 500 + 400,:)];
    testX = [testX;feature((i - 1) * 500 + 401:i * 500,:)];
    trainY = [trainY;y((i - 1) * 500 + 1:(i - 1) * 500 + 400,:)];
    testY = [testY;y((i - 1) * 500 + 401:i * 500,:)];
end

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