#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Convert raw data to data for learning'''

__author__ = 'Chaoliang Zhong'

from decimal import Decimal as D
import math
import os
import csv
import random

# the rate for stopping loss
stop_loss_rate = 0.10

# the rate for taking profit
take_profit_rate = 0.10

# the maximum holding dates of the stock
days = 30

# a cache for stopping loss
lose_cache = 0.005

# the number of features
num_feature = 108

# the number of lines of ignored data
num_ignore = 35

# root directory of data
root_dir = '../../'

def round_float(g, pos=2):
    if g<0:
        f = -g
    else:
        f = g
    p1 = pow(D('10'), D(str(pos+1)))
    last = D(str(int(D(str(f))*p1)))%D('10')
    p = pow(D('10'), D(str(pos)))
    if last >= 5:
        result = float(math.ceil(D(str(f))*p)/p)
    else:
        result = float(math.floor(D(str(f))*p)/p)
    if g<0:
        return -result
    else:
        return result

def process_file(path):
    data = []
    content = []
    with open(path, 'r') as f:
        i = 0
        # ignore the first *num_ignore* lines of data
        for line in f.readlines():
            line_data = line.strip().split('\t')
            if i > num_ignore and len(line_data) == num_feature:
                item = []
                item.append(line_data[0].strip())
                for j in range(1, len(line_data)):
                    if line_data[j].strip() == '':
                        item.append(None)
                    else:
                        item.append(float(line_data[j].strip()))
                content.append(item)
            i += 1

    for i in range(len(content)):
        open_price = content[i][1]
        high_price = content[i][2]
        low_price = content[i][3]
        close_price = content[i][4]
        volume = content[i][5]
        increase_amount = content[i][6]

        winer = []

        for k in range(31):
            if content[i][7+k] is None:
                break
            else:
                winer.append(content[i][7+k])

        if len(winer) < 31:
            continue

        increase_days = content[i][38]

        increase = []

        for k in range(31):
            increase.append(content[i][39+k])

        turnover = []

        for k in range(31):
            turnover.append(content[i][70+k])

        buy = int(content[i][101])
        follow = int(content[i][102])

        index = []

        for k in range(5):
            if content[i][103+k] is None:
                break
            else:
                index.append(content[i][103+k])

        if len(index) < 5:
            continue

        if i + days + 1 < len(content) and follow == 1:
            new_open_price = content[i+1][1]
            win_price = round_float(new_open_price * (1+take_profit_rate))
            lose_price = round_float(new_open_price * (1-stop_loss_rate+lose_cache))
            success = 0
            for j in range(i+2, i+ days + 2):
                new_high_price = content[j][2]
                new_low_price = content[j][3]
                if new_low_price <= lose_price:
                    success = 0
                    break
                if new_high_price >= win_price:
                    success = 1
                    break
            if success == 0 or success == 1:
                data_item = []
                data_item.extend(turnover)
                data_item.extend(winer)
                data_item.extend(increase)
                data_item.extend(index)
                data_item.append(increase_days)
                data_item.append(increase_amount)
                data_item.append(success)
                data.append(data_item)

    return data

def process_folder(path):
    path = os.path.abspath(path)
    data = []
    for x in os.listdir(path):
        new_path = os.path.join(path, x)
        if os.path.isdir(new_path):
            data[0:0] = process_folder(new_path)
        elif os.path.isfile(new_path) and os.path.splitext(x)[1]=='.txt':
            print(new_path)
            data[0:0] = process_file(new_path)
    return data

def write_csv(path, data):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


data = process_folder(root_dir+'/data/data_buy_follow_index_1/raw')
random.shuffle(data)
write_csv(root_dir+'/data/data_buy_follow_index_1/csv/ten_percent/thirty_days/data.csv', data)

title = []
for k in range(31):
    title.append('turnover' + str(k))
for k in range(31):
    title.append('winer' + str(k))
for k in range(31):
    title.append('increase' + str(k))
for k in range(5):
    title.append('index' + str(k))
title.append('increase_days')
title.append('increase_amount')
title.append('success')
data.insert(0, title)
write_csv(root_dir+'/data/data_buy_follow_index_1/csv/ten_percent/thirty_days/data_weka.csv', data)


