function result = DecisionTree(trainX,trainY,testX,testY)
tic;
x = trainX;
y = num2str(trainY);
y = mat2cell(y,ones(size(trainY)));
% 构造分类决策树
t = classregtree(x,y);
% 决策树分类
result = eval(t,testX);
% 获得分类结果
result = cell2mat(result);
result = str2num(result);
rate = sum(result == testY) / length(testY);
fprintf('测试样本的正确率为\n %d%%\n', round(rate*100));
% 显示分类结果
treedisp(t);
ti = toc;
fprintf('完成分类共耗时 %f sec\n', ti);