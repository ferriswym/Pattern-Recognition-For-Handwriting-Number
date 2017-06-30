function result = AdaBoost(trainX,trainY,testX,testY)
tic;
ada = fitensemble(trainX,trainY,'AdaBoostM2',100,'Tree');
result = predict(ada,testX);
rate = sum(result == testY) / length(testY);
fprintf('������������ȷ��Ϊ\n %d%%\n', round(rate*100));
ti = toc;
fprintf('��ɷ��๲��ʱ %f sec\n', ti);