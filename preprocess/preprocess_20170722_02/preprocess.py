#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Convert raw data to data for learning'''

__author__ = 'Chaoliang Zhong'

from decimal import Decimal as D
import math
import os
import csv
import random
import json

# the rate for stopping loss
stop_loss_rate = 0.10

# the rate for taking profit
take_profit_rate = 0.10

# the maximum holding dates of the stock
hold_days = 30

# days for LSTM analysis
old_days = 30

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
    label = []
    content = []

    # 股票属于沪市、深市或创业板
    code_info = [0, 0, 0]
    with open(path, 'r') as f:
        i = 0
        # ignore the first *num_ignore* lines of data
        for line in f.readlines():
            if i == 0:
                items = line.strip().split(' ')
                print(items[len(items)-1])
                code_start = items[len(items)-1][1]
                print(code_start)
                if code_start == '0':
                    code_info = [1, 0, 0]
                elif code_start == '6':
                    code_info = [0, 1, 0]
                else:
                    code_info = [0, 0, 1]
            line_data = line.strip().split('\t')
            if i > num_ignore and len(line_data) == num_feature:
                item = []
                item.append(line_data[0].strip())
                for j in range(1, len(line_data)):
                    if line_data[j].strip() == '':
                        item.append(-1)
                    else:
                        item.append(float(line_data[j].strip()))
                content.append(item)
            i += 1

    for i in range(len(content)):

        buy = int(content[i][101])
        follow = int(content[i][102])

        if i + hold_days + 1 < len(content) and follow == 1:
            new_open_price = content[i+1][1]
            win_price = round_float(new_open_price * (1+take_profit_rate))
            lose_price = round_float(new_open_price * (1-stop_loss_rate+lose_cache))
            success = 0
            for j in range(i+2, i+ hold_days + 2):
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
                for k in range(old_days):
                    item = []
                    l = i-old_days+k+1
                    if l < 1:
                        break
                    open_price = content[l][1]
                    high_price = content[l][2]
                    low_price = content[l][3]
                    close_price = content[l][4]

                    # 最低价与开盘价之比
                    low_open = low_price/open_price;
                    item.append(low_open)
                    # 收盘价与开盘价之比
                    close_open = close_price/open_price;
                    item.append(close_open)
                    # 最高价与开盘价之比
                    high_open = high_price/open_price;
                    item.append(high_open)
                    # 收盘价与前一日收盘价之比
                    close_pre = close_price/content[l-1][4];
                    item.append(close_pre)
                    # 重心是否提高
                    up_or_not = int(close_price+open_price>content[l-1][4]+content[l-1][1])
                    item.append(up_or_not)
                    # 连涨天数
                    increase_days = content[l][38]
                    item.append(increase_days)
                    # 重心最大连续提高幅度
                    increase_amount = content[l][6]
                    item.append(increase_amount)
                    # 换手率
                    turnover = content[l][70]
                    item.append(turnover)
                    # 获利盘比例
                    winner = content[l][7]
                    item.append(winner)

                    # 沪指是否大于5日均线、10日均线、20日均线、30日均线、60日均线
                    for m in range(5):
                        item.append(content[l][103+m])
                    
                    item.extend(code_info)

                    data_item.append(item)
                if len(data_item) == old_days:
                    data.append(data_item)
                    label.append(success)
    return data, label

def process_folder(path):
    path = os.path.abspath(path)
    data = []
    label = []
    for x in os.listdir(path):
        new_path = os.path.join(path, x)
        if os.path.isdir(new_path):
            x, y = process_folder(new_path)
            data[0:0] = x
            label[0:0] = y
        elif os.path.isfile(new_path) and os.path.splitext(x)[1]=='.txt':
            print(new_path)
            x, y = process_file(new_path)
            data[0:0] = x
            label[0:0] = y
    return data, label

def write_data(path, data):
    with open(path, 'w', newline='') as f:
        f.write(json.dumps(data))

x, y = process_folder(root_dir+'/data/data_buy_follow_index_1/raw')
write_data(root_dir+'/data/data_buy_follow_index_1/json/data.json', [x, y])