function result = AdaBoost(trainX,trainY,testX,testY)
tic;
ada = fitensemble(trainX,trainY,'AdaBoostM2',100,'Tree');
result = predict(ada,testX);
rate = sum(result == testY) / length(testY);
fprintf('测试样本的正确率为\n %d%%\n', round(rate*100));
ti = toc;
fprintf('完成分类共耗时 %f sec\n', ti);