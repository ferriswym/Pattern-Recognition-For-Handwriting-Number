function result = BayesNaive(trainX,trainY,testX,testY)
tic;            %����ʱ��
model = NaiveBayes.fit(trainX,trainY);
result = model.predict(testX);
rate = sum(result == testY) / length(testY);
fprintf('������������ȷ��Ϊ\n %d%%\n', round(rate*100));
ti = toc;
fprintf('��ɷ��๲��ʱ %f sec\n', ti);