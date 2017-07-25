import xgboost as xgb

# train_data / all_data = 0.96, without cross validation
# positive_threshold = 0.50 => accuracy = 4396/6897 = 0.6373, precision = 1663/2651 = 0.62
# positive_threshold = 0.67 => accuracy = 4106/6897 = 0.5953, precision = 492/599 = 0.82
# positive_threshold = 0.80 => accuracy = 3889/6897 = 0.5638, precision = 176/184 = 0.95
# positive_threshold = 0.90 => accuracy = 3764/6897 = 0.5457, precision = 44/45 = 0.97
# positive_threshold = 0.95 => accuracy = 3724/6897 = 0.5399, precision = 3/3 = 1.00


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

positive_threshold_list = [0.50, 0.67, 0.80, 0.90, 0.95]

for positive_threshold in positive_threshold_list:
	print('positive_threshold: ' + str(positive_threshold))
	num_correct = sum(1 for i in range(len(preds)) if int(preds[i]>positive_threshold)==labels[i])
	num_pred = len(preds)
	num_error = num_pred - num_correct
	print ('error=%d/%d=%f' % (num_error, num_pred, num_error /float(num_pred)))

	print ('accuracy=%d/%d=%f' % ( num_correct, num_pred, num_correct /float(num_pred)))

	num_true_positive = sum(1 for i in range(len(preds)) if int(preds[i]>positive_threshold)==labels[i] and labels[i]==1)
	num_positive_pred = sum(1 for i in range(len(preds)) if preds[i]>positive_threshold)
	print ('precision=%d/%d=%f' % ( num_true_positive, num_positive_pred, num_true_positive /float(num_positive_pred)))
	print('')