"""
AnomalyDetection ML code
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import datetime
from scipy import stats
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
import os
import json
import warnings
from pytz import timezone
import argparse
import logging
import logging.handlers
import boto3
import yaml



est = timezone('US/Eastern')
warnings.filterwarnings("ignore")


def log(msg):
    # logging.info(msg)
    print(msg)


def read_csv_file(input_file_path):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(input_file_path)
        log("Shape of the data in file {} is {}".format(input_file_path, df.shape))
        if df.shape[0] == 0:
            log("No data in file {}".format(input_file_path))
    except Exception as e:
        log("Issue while reading data at {} \n{}".format(input_file_path, e))
    return df


def standardize_date_col(dataframe, date_col):
    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format='%Y-%m-%d', errors='coerce')
    # Sort the data by the 'Date' column in ascending order
    dataframe = dataframe.sort_values(date_col, ignore_index=True)
    return dataframe


def get_lag_columns(data, groupby_cols, lag_columns, lags):
    # Group the data by 'State' and perform lag shifting within each group
    grouped = data.groupby(groupby_cols)

    # Create lag features within each group
    for lag in lags:
        for col in lag_columns:
            data[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
            data[f'{col}_lag_{lag}'] = data[f'{col}_lag_{lag}'].bfill()
    return data


def get_date_features(data, date_col):
    def week_of_month(date):
        # Get the first day of the month
        first_day = date.replace(day=1)
        # Calculate the adjusted day of the week (0=Monday, ..., 6=Sunday)
        adjusted_dom = (first_day.weekday() + 1) % 7
        # Calculate the week of the month
        week_of_month = (date.day + adjusted_dom - 1) // 7 + 1
        return week_of_month
    data[date_col] = pd.to_datetime(data[date_col])
    data['Weekend'] = data[date_col].dt.weekday.isin([5, 6]).map({True: 1, False: 0})
    data['Week_Of_Month'] = data[date_col].map(week_of_month)
    # Create time-based features
    data['Day_of_Week'] = data[date_col].dt.dayofweek  # Day of the week (0: Monday, 1: Tuesday, ..., 6: Sunday)
    data['Month'] = data[date_col].dt.month  # Month of the year (1 to 12)
    data['Quarter'] = data[date_col].dt.quarter  # Quarter of the year (1 to 4)
    data['Year'] = data[date_col].dt.year  # Year
    return data


def get_decomposed_values(data, filter_values):
    decomposed_values_df = pd.DataFrame()
    for filter_value in filter_values:
        df_sub_filter = data[data[filter_col] == filter_value].reset_index(drop=True).copy()
        group_df = df_sub_filter[[date_col, target]].groupby([date_col])[target].apply(sum).reset_index()

        group_df[date_col] = pd.to_datetime(group_df[date_col], format="%Y-%m-%d")

        group_df.sort_values(by=date_col, inplace=True)

        # STL - For Trend Analysis
        salesValues = list(group_df[target])
        dates = list(group_df[date_col])

        filterData = pd.DataFrame(salesValues, index=dates)
        filterData.index = pd.to_datetime(filterData.index, format="%d-%m-%Y")
        filterData.columns = ['Value']
        # filterData = filterData.asfreq('W-FRI').interpolate(method='linear')
        filterData = filterData.asfreq('D').interpolate(method='linear')

        padding_length = 4
        data_padded = np.pad(filterData.Value.tolist(), (padding_length, padding_length), 'edge')

        # STL Decomp of Data
        result = seasonal_decompose(data_padded, model='additive', period=7)
        trend = result.trend[padding_length:-padding_length]
        seasonal = result.seasonal[padding_length:-padding_length]
        residual = result.resid[padding_length:-padding_length]

        filterData[filter_col] = filter_value
        filterData['trend'] = trend
        filterData['seasonal'] = seasonal
        filterData['residual'] = residual

        decomposed_values_df = pd.concat([decomposed_values_df, filterData])

    decomposed_values_df[date_col] = decomposed_values_df.index
    decomposed_values_df = decomposed_values_df.reset_index(drop=True)
    return decomposed_values_df

def split_train_data(data):
    train_size = int(np.floor(data.shape[0] * 0.7))
    val_size = int(np.floor(data.shape[0] * 0.7) - train_size)
    print(train_size, val_size, train_size + val_size, data.shape[0],
          "\ntrain_size, val_size, train_size + val_size, data.shape[0]")
    train_data = df_train.iloc[:train_size]
    val_data = df_train.iloc[train_size:train_size + val_size]
    return train_data, val_data


def train_XGBmodel(X_train, y_train, X_val, y_val):
    # Define a narrower set of hyperparameters
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.2,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
    }
    # Define a narrower set of hyperparameters
    # Train the XGBoost model with the reduced set of hyperparameters
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)

    # Evaluate the model performance on the validation set
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    validation_report = classification_report(y_val, y_val_pred)

    print("Validation Accuracy:", validation_accuracy)
    print("Validation Report:")
    print(validation_report)
    return model


def train_model(data, drop_for_training, ):
    train_data, val_data = split_train_data(data)
    print(train_data.shape, val_data.shape, val_data['Anomaly'].value_counts())

    # Split the data into features and target variable
    X_train = train_data.drop(drop_for_training, axis=1)
    y_train = train_data[train_target]

    X_val = val_data.drop(drop_for_training, axis=1)
    y_val = val_data[train_target]

    print("X_train.shape, X_val.shape", X_train.shape, X_val.shape)
    print("X_train.columns", X_train.columns)
    model = train_XGBmodel(X_train, y_train, X_val, y_val)
    return model

def save_model(model):
    # save the model to S3
    print("in save model")
    return

if __name__ == "__main__":
    print("Starting ML Pipeline")
    # Collect all command line arguments.
    print("Collecting Command Line Arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-model", type=str, default='No')
    args, _ = parser.parse_known_args()
    train_model_flag = args.train_model
    if train_model_flag == 'yes':
        parser.add_argument("--train-filename", type=str)
        parser.add_argument("--inference-filename", type=str)
        parser.add_argument("--result-file-path", type=str)
        parser.add_argument("--processing-job-names", type=str)
        args, _ = parser.parse_known_args()
        train_file = args.train_filename
        inference_file = args.inference_filename
        processing_job_names = args.processing_job_names
        result_file_path = args.result_file_path
    else:
        parser.add_argument("--inference-filename", type=str)
        parser.add_argument("--train-filename", type=str)
        parser.add_argument("--result-file-path", type=str)
        parser.add_argument("--processing-job-names", type=str)
        args, _ = parser.parse_known_args()
        train_file = args.train_filename  # "na"
        inference_file = args.inference_filename
        processing_job_names = args.processing_job_names
        result_file_path = args.result_file_path
    print("Received arguments {}".format(args))
    print("Command Line Arguments Collected")
    print("Calling Pipeline code")
    print("train_file: ", train_file)
    print("inference_file: ", inference_file)
    print("Processing job names ", processing_job_names)
    rootPath = "/opt/ml/processing"
    local_train_file = os.path.join(rootPath + "/input/train/" + train_file.rsplit('/', 1)[1])
    local_inference_file = os.path.join(rootPath + "/input/inference/" + inference_file.rsplit('/', 1)[1])
    print("local_inference_file, local_train_file:", local_inference_file, local_train_file, )
    df_train = read_csv_file(local_train_file)
    df_inference = read_csv_file(local_inference_file)

    print(df_train.columns, df_inference.columns)

    ## Start training
    date_col = 'time_period'
    target = 'value'
    filter_col = 'key1'
    groupby_cols = ['key1']
    lag_columns = [target]
    lags = [1, 2, 4]  # Example lag values of 1 week and 2 weeks
    drop_for_training = ['time_period', 'upper_value', 'lower_value', 'anomaly', 'train_flag', 'feedback_flag']
    train_target = 'anomaly'
    df_train = standardize_date_col(df_train, date_col)
    df_train = get_lag_columns(df_train, groupby_cols, lag_columns, lags)
    df_train = get_date_features(df_train, date_col)
    filter_values = df_train[filter_col].unique()
    print("filter_values", filter_values)

    decomposed_values_df = get_decomposed_values(df_train, filter_values)
    print("decomposed_values_df:", decomposed_values_df.sample(3))
    df_train = df_train.merge(decomposed_values_df[[date_col, filter_col, "trend", "seasonal", "residual"]],
                      on=[date_col, filter_col], how="inner")
    print(df_train.shape, "shape of train data after merging with decomposed values.")
    # Encode categorical columns
    encoder = LabelEncoder()
    df_train[filter_col] = encoder.fit_transform(df_train[filter_col])
    df_train = df_train.reset_index(drop=True)
    df_train[target + "_ma"] = round(df_train[target].ewm(
        alpha=0.5, adjust=False).mean(), 2)

    model = train_model(df_train, drop_for_training)
    save_model(model)



    print("Sagemaker processing job complete")
