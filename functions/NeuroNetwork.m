function result = NeuroNetwork(trainX,trainY,testX,testY)
tic;
spread = 0.1;
%������ѵ��������
net = newpnn(trainX',ind2vec(trainY'),spread);          
ti = toc;
fprintf('��������ģ�͹���ʱ %f sec\n', ti);

lab0 = net(testX');
result = vec2ind(lab0);
rate = sum(result' == testY) / length(testY);
fprintf('������������ȷ��Ϊ\n %d%%\n', round(rate*100));