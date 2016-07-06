%%  error
[tp ,tn ,fp ,fn] = predict_and_return(data , theta);
%train stats
%[tp tn fp fn]
Train_accuracy = (tp+tn) / (tp+fp+tn+fn)
Train_precision = tp / (tp+fp)
Train_recall = tp / (tp+fn)
Train_F1_score = (2*Train_precision*Train_recall)/(Train_precision+Train_recall)


test2 = data(180000:end,:);
[tp ,tn ,fp ,fn] = predict_and_return(test2,theta);
%train stats
%[tp tn fp fn]
Test2_accuracy = (tp+tn) / (tp+fp+tn+fn)
Test2_precision = tp / (tp+fp)
Test2_recall = tp / (tp+fn)
Test2_F1_score = (2*Test2_precision*Test2_recall)/(Test2_precision+Test2_recall)




