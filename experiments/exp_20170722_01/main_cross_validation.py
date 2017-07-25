import xgboost as xgb

# train_data / all_data = 0.8, cross validation
# folder_1
# positive_threshold = 0.50 => accuracy = 21940/34484 = 0.6362, precision = 8480/13484 = 0.6288
# positive_threshold = 0.67 => accuracy = 20481/34484 = 0.5939, precision = 2574/3131 = 0.8221
# positive_threshold = 0.80 => accuracy = 19242/34484 = 0.5579, precision = 825/872 = 0.9461
# positive_threshold = 0.90 => accuracy = 18653/34484 = 0.5409, precision = 190/191 = 0.9947
# positive_threshold = 0.95 => accuracy = 18485/34484 = 0.5360, precision = 21/21 = 1.0000
# folder_2
# positive_threshold = 0.50 => accuracy = 22003/34484 = 0.6380, precision = 8422/13257 = 0.6352
# positive_threshold = 0.67 => accuracy = 20452/34484 = 0.5930, precision = 2608/3180 = 0.8201
# positive_threshold = 0.80 => accuracy = 19216/34484 = 0.5572, precision = 842/884 = 0.9524
# positive_threshold = 0.90 => accuracy = 18639/34484 = 0.5405, precision = 228/233 = 0.9785
# positive_threshold = 0.95 => accuracy = 18441/34484 = 0.5347, precision = 25/25 = 1.0000
# folder_3
# positive_threshold = 0.50 => accuracy = 22112/34484 = 0.6412, precision = 8590/13492 = 0.6366
# positive_threshold = 0.67 => accuracy = 20565/34484 = 0.5963, precision = 2743/3345 = 0.8200
# positive_threshold = 0.80 => accuracy = 19287/34484 = 0.5593, precision = 916/969 = 0.9453
# positive_threshold = 0.90 => accuracy = 18606/34484 = 0.5395, precision = 185/188 = 0.9840
# positive_threshold = 0.95 => accuracy = 18438/34484 = 0.5346, precision = 14/14 = 1.0000
# folder_4
# positive_threshold = 0.50 => accuracy = 22106/34484 = 0.6410, precision = 8488/13183 = 0.6438
# positive_threshold = 0.67 => accuracy = 20374/34484 = 0.5908, precision = 2555/3049 = 0.8379
# positive_threshold = 0.80 => accuracy = 19123/34484 = 0.5545, precision = 848/886 = 0.9571
# positive_threshold = 0.90 => accuracy = 18504/34484 = 0.5365, precision = 194/197 = 0.9847
# positive_threshold = 0.95 => accuracy = 18328/34484 = 0.5314, precision = 15/15 = 1.0000
# folder_5
# positive_threshold = 0.50 => accuracy = 21919/34484 = 0.6356, precision = 8332/13214 = 0.6305
# positive_threshold = 0.67 => accuracy = 20450/34484 = 0.5930, precision = 2495/3009 = 0.8291
# positive_threshold = 0.80 => accuracy = 19252/34484 = 0.5582, precision = 816/849 = 0.9611
# positive_threshold = 0.90 => accuracy = 18629/34484 = 0.5402, precision = 161/162 = 0.9938
# positive_threshold = 0.95 => accuracy = 18477/34484 = 0.5358, precision = 8/8 = 1.0000


# read in data
dtrain = xgb.DMatrix('../../data/data_20170722_01/train_data_f5.txt')
dtest = xgb.DMatrix('../../data/data_20170722_01/test_data_f5.txt')

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