function result = SVMDecision(trainX,trainY,testX,testY)
tic;            %开启时钟
model = svmtrain(trainY,trainX);
[result] = svmpredict(testY, testX, model);
ti = toc;
fprintf('完成分类共耗时 %f sec\n', ti);