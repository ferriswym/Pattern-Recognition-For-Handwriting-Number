function result = RandomForest(trainX,trainY,testX,testY)
tic;
nTree = 500;
y = num2str(trainY);
y = mat2cell(y,ones(size(trainY)));
forest = TreeBagger(nTree,trainX,y);
temp = predict(forest,testX);
result = zeros(size(temp));
for i = 1:length(temp) 
    result(i) = str2num(cell2mat(temp(i)));
end
rate = sum(result == testY) / length(testY);
fprintf('������������ȷ��Ϊ\n %d%%\n', round(rate*100));
ti = toc;
fprintf('��ɷ��๲��ʱ %f sec\n', ti);