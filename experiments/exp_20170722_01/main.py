import xgboost as xgb

# Accuracy on test set: 63%

# read in data
dtrain = xgb.DMatrix('../../data/data_20170722_01/train_data.txt')
dtest = xgb.DMatrix('../../data/data_20170722_01/test_data.txt')

# specify parameters via map, definition are same as c++ version
param = {'max_depth':22, 'eta':0.1, 'silent':0, 'objective':'binary:logistic','min_child_weight':3,'gamma':14 }

# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 60
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

print ('correct=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)==labels[i]) /float(len(preds))))

print ('precision=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)==labels[i] and labels[i]==1) /float(sum(1 for i in range(len(preds)) if preds[i]>0.5))))