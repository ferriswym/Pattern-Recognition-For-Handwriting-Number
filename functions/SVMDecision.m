function result = SVMDecision(trainX,trainY,testX,testY)
tic;            %����ʱ��
model = svmtrain(trainY,trainX);
[result] = svmpredict(testY, testX, model);
ti = toc;
fprintf('��ɷ��๲��ʱ %f sec\n', ti);