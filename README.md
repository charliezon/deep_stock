# Deep Stock

This project aims to build effective stock selection models based on machine learning tools such as Keras, Tensorflow, XGBoost, etc., for Chinese stock market participants.

## Data

### Raw data

Raw data are colllected using TDX (http://www.tdx.com.cn/) client with a tool that can press buttons of keyboard and mouse automatically according to a recorded script.

### Data for learning

Raw data are converted to csv format for learning. It can be downloaded at https://pan.baidu.com/s/1pLnxvab.

### Features

Basically, the features include turnover, increase rate, winner rate of recent thirty days and so on. The set of features will vary depending on the specific experiment settings.

## Stock trade strategy

Suppose we set the same rate for stopping loss and taking profit, then when the success rate is larger than fifty percent, we can profit. Therefore, our goal is to find the models with as high success rate as possible in this condition.

## Dependencies

* Keras 2.0.6
* XGBoost 0.6

## Usage

Clone this repository, download and extract data to corresponding folders, and execute main.py in corresponding experiment folders.
