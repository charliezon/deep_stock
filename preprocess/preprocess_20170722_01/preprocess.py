#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Chaoliang Zhong'

import csv

root_dir = '../../'

# convert from csv to libsvm format
def convert(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data_item = []
            data_item.append(row[len(row)-1]);
            for i in range(len(row)-1):
                data_item.append(str(i+1) + ':' + row[i])
            data.append(' '.join(data_item))
    return data

def write_file(path, data):
    with open(path, 'w') as f:
        f.write(data)

data = convert(root_dir + '/data/data_buy_follow_index_1/csv/ten_percent/thirty_days/data.csv')

train_len = int(len(data)*0.96)
train_data = data[0:train_len]
test_data = data[train_len:]

write_file(root_dir + '/data/data_buy_follow_index_1/csv/ten_percent/thirty_days/train_data.txt', '\n'.join(train_data))
write_file(root_dir + '/data/data_buy_follow_index_1/csv/ten_percent/thirty_days/test_data.txt', '\n'.join(test_data))