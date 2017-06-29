function result = BayesNaive(trainX,trainY,testX,testY)
tic;            %开启时钟
model = NaiveBayes.fit(trainX,trainY);
result = model.predict(testX);
rate = sum(result == testY) / length(testY);
fprintf('测试样本的正确率为\n %d%%\n', round(rate*100));
ti = toc;
fprintf('完成分类共耗时 %f sec\n', ti);