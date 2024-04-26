'''
    Title: Anomaly Detection Pipeline Script
    Use Case: TnC data

    Created by: Nitish Gade
    Updated by: Geeta Phatak & Rahul Sivankutty

    Description: This code presents the machine learning pipeline for anomaly detection.
    It includes steps for labeling, preprocessing,
                 training, evaluating, and making inferences from data.

    Date: 07/24/2021
    03/11/2022: Updating code to train an XGBoost model on TnC Completeness data
                Cleaning up commented code -Geeta
    12/09/2022: Adding regular pipelines to TnC-completeness and inactivating/separating the snapshot_date based pipeline.
'''

import pandas as pd
import os
import json
import numpy as np
import warnings
import datetime
import time
from pytz import timezone
import ast
import argparse
import logging
import logging.handlers
# import boto3
import yaml
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import confusion_matrix, roc_curve
import json
# import psycopg2
# from imblearn.combine import SMOTEENN  # TODO
# from imblearn.under_sampling import EditedNearestNeighbours  # TODO
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from ThymeBoost import ThymeBoost as TB
import pandas.io.sql as sqlio

# from cryptography.fernet import Fernet ## TODO

est = timezone('US/Eastern')
warnings.filterwarnings("ignore")


class AnomalyDetectionPipeline:
    """  Main code for preprocessing, training and inference of data  """

    def __init__(self, path, train_file, shift_n, input_dict, debugMode, inference_data_flag):
        self.data = pd.DataFrame() ## Geeta this should not create a prob
        self.train_file = train_file
        self.job_start_time = datetime.datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
        ''' Read environment variables '''
        self.log("Read environment variables")
        self.ENDPOINT = os.environ.get('host')
        self.PORT = os.environ.get('port')
        self.USR = os.environ.get('username')
        self.DBNAME = os.environ.get('dbname')
        self.PASSWORD = os.environ.get('password')
        self.START_TIME = os.environ.get('start_time')
        self.END_TIME = ''
        self.JOB_RUN_ID = os.environ.get('job_run_id')
        self.SnsRoleARN = os.environ.get('SnsRoleArn')
        self.SnsRoleArnDeveloper = os.environ.get('SnsRoleArnDeveloper')
        self.model_algo = ''
        self.tnc_completeness_threshold = ''
        self.debugMode = debugMode
        self.inference_data_flag = inference_data_flag
        self.shift_n = shift_n
        self.file_name = path
        self.feedback_data = pd.DataFrame()
        self.saved_inference_df = pd.DataFrame()
        self.env = os.environ.get('env')
        self.dq_metric = os.environ.get('dq_metric')
        self.dashboard_name = os.environ.get('dashboard_name')
        self.RedshiftHost = os.environ.get('RedshiftHost')
        self.RedshiftPort = os.environ.get('RedshiftPort')
        self.RedshiftDbname = os.environ.get('RedshiftDbname')
        self.RedshiftUsername = os.environ.get('RedshiftUsername')
        self.RedshiftPassword = os.environ.get('RedshiftPassword')
        self.prev_target_col = ''
        self.current_target_col = ''

        if debugMode:
            # Debug mode to run and test the code in local

            self.PIPELINE = input_dict["PIPELINE"]
            self.TRIAL_TYPE =  input_dict["TRIAL_TYPE"] # "None"  #
            self.TABLE_NAME = input_dict["TABLE_NAME"]  # aggr_dq_mdm_agility_validity_cna"
            self.TARGET = input_dict["TARGET"]  # "agility_validity_cna"
            self.SNAPSHOT_DATE = input_dict["SNAPSHOT_DATE"]
            self.YAML_CONFIG_FILE_NAME = input_dict["YAML_CONFIG_FILE_NAME"]
            self.DATA_METRIC =input_dict["DATA_METRIC"] #'("COMPLETENESS NOOP BEACONING")' # '("Validity - CNA Records with Valid Mailing Address")'  # "('Validity - CNA Records with Valid Mailing Address')"
            self.TRAIN_FILE_PATH = path
        else:
            self.PIPELINE = os.environ.get('pipeline_name')
            self.TRIAL_TYPE = os.environ.get('trial_type')
            self.DATA_METRIC = os.environ.get('data_metric')
            self.TABLE_NAME = os.environ.get('table_name')
            self.TARGET = os.environ.get('target')
            self.SNAPSHOT_DATE = os.environ.get('snapshot_date')
            self.YAML_CONFIG_FILE_NAME = os.environ.get('yaml_config')
            self.TRAIN_FILE_PATH = os.environ.get('train_file_path')
            self.S3Bucket = os.environ.get('s3_bucket_name')

        self.log("Environment variables read - complete \nclass AnomalyDetectionPipeline __init__() compelete")
        # self.log("data_metric value is", self.DATA_METRIC)

        self.main()

    def log(self, msg):
        # logging.info(msg)
        print(msg)

    def set_datetime_filename(self):
        """  have file names with current date and time  """
        current_date_time = datetime.datetime.now(est)
        current_year = str(current_date_time.year)
        current_month = str(current_date_time.month)
        current_day = str(current_date_time.day)
        current_hour = str(current_date_time.hour)
        current_minute = str(current_date_time.minute)
        current_second = str(current_date_time.second)
        current_microsec = str(current_date_time.microsecond)
        self.datetime_filename = current_year + "-" + current_month + "-" \
                                 + current_day + "-" + str(int(current_hour)) \
                                 + '-' + current_minute + "-" + current_second \
                                 + "-" + current_microsec

    def get_local_file_paths(self):
        """ local files path where output of the program will be stored in the container."""
        self.rootPath = "/opt/ml/processing"
        self.inputPath = self.rootPath + "/input"
        self.metadataPath = self.rootPath + "/metadata"
        self.configPath = self.rootPath + "/yamlconfig"
        self.config_localFilename = self.configPath + "/" + self.YAML_CONFIG_FILE_NAME
        if self.debugMode:
            with open(self.YAML_CONFIG_FILE_NAME) as f:
                self.config = yaml.safe_load(f)
        else:
            with open(self.config_localFilename) as f:
                self.config = yaml.safe_load(f)

        self.table_parameters = self.config[self.TARGET]
        self.labelPath = self.rootPath + "/label"
        self.processedPath = self.rootPath + "/processed"
        self.trainPath = self.rootPath + "/input"
        self.testPath = self.rootPath + "/test"
        self.modelPath = self.rootPath + "/model"
        self.modelMetadataPath = self.rootPath + "/model_metadata"
        self.evalPath = self.rootPath + "/evaluation"
        self.predictPath = self.rootPath + "/predicted"  # predicted contains inference data (unseen, new data) that has already been preprocessed.
        self.logPath = self.rootPath + "/logs"
        self.resultPath = self.rootPath + "/results"  # results contains inference data (unseen, new data) that has not been preprocessed but has a column for predictions.
        self.inferencePath = self.rootPath + "/inference"
        self.pretrainedModelPath = self.rootPath + "/pretrained_model"
        self.processingJobDetailsPath = self.rootPath + "/processingJobDetails"
        self.feedbackDataPath = self.rootPath + "/feedback"
        self.feedbackDataPathOutput = self.rootPath + "/feedback_output"
        self.inferenceDataPathOutput = self.rootPath + "/saved_inference_out"
        self.rawInferencePathOut = self.rootPath + "/raw_inference_out"
        self.check_directories()
        self.confirm_trial_type()

    def confirm_trial_type(self):
        try:
            alert_id = self.table_parameters['trial_type'][self.TRIAL_TYPE]['alert_id']
        except Exception as e:
            print("trial_type is not right, setting it straight", e)
            if self.TRIAL_TYPE == 'new-sold': self.TRIAL_TYPE = 'NEW_SOLD'
            elif self.TRIAL_TYPE == 'used-sold': self.TRIAL_TYPE = 'USED_SOLD'
            elif self.TRIAL_TYPE == 'service-lane': self.TRIAL_TYPE = 'SERVICE_LANE'
            else: self.TRIAL_TYPE = 'all'

    def check_directories(self):
        """
        This function checks if the output paths that will be needed by the Python script exist in the container folder structure.
        If they do not exist, then create those paths.
        """
        dirlist = [self.labelPath, self.processedPath, self.trainPath, self.testPath, self.modelPath, self.evalPath,
                   self.predictPath, self.resultPath, self.logPath, self.pretrainedModelPath, self.modelMetadataPath,
                   self.processingJobDetailsPath, self.feedbackDataPath, self.feedbackDataPathOutput,
                   self.inferenceDataPathOutput, self.rawInferencePathOut]
        for path_ in dirlist:
            if not os.path.exists(path_):
                self.log('path doesnt exists: ' + path_)
                os.makedirs(path_)

    def create_filenames(self):
        """
        Add comments here too
        """
        self.set_datetime_filename()
        self.log('self.datetime_filename' + self.datetime_filename)
        labeled_data = self.TARGET + "-labeled_data.csv"  # + self.datetime_filename + ".csv"
        train_f = "train_features_" + self.datetime_filename +"_" + str(self.shift_n) + "days.csv"
        train_l = "train_labels_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        test_f = "test_features_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        test_l = "test_labels_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        processed = "processed_data_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        processed_inf = "inference_processed_data_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        model_file = str(self.shift_n) + "days_model.pkl"
        ev = "evaluation_" + self.datetime_filename + "_" + str(self.shift_n) + "days.json"
        model_details_file = str(self.shift_n) + "days_model_details.json"
        processed_predicted = "processed_predicted_data_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        original_predicted = "original_predicted_data_" + self.datetime_filename + "_" + str(self.shift_n) + "days.csv"
        processing_job_details = "processing_job_details_" + self.datetime_filename + "_" + str(self.shift_n) + "days.json"
        inference_file_save = self.TARGET + "_" + str(self.shift_n) + "days" + "-inference.csv"
        feedback_file = self.TARGET + "_" + str(self.shift_n) + "days" + "-feedback.csv"
        feedback_history_file = self.TARGET + "_" + str(self.shift_n) + "days" + "-feedback-history.csv"
        feedback_trend = self.TARGET + "_" + str(self.shift_n) + "days" + "-trend.csv"

        self.filenames = {'labeled_data': labeled_data, 'train_features': train_f,
                          'train_labels': train_l, 'test_features': test_f, 'test_labels': test_l,
                          'processed_data': processed, 'inference_processed_data': processed_inf,
                          'model_file': model_file, 'evaluation': ev, 'processed_predicted_data': processed_predicted,
                          'original_predicted_data': original_predicted, 'model_details_file': model_details_file,
                          'processing_job_details': processing_job_details, 'inference_save': inference_file_save,
                          'feedback_file':feedback_file,'feedback_history_file':feedback_history_file, 'feedback_trend':feedback_trend
                          }

    def save_file(self, df, location, msg):
        df.to_csv(location, header=True, index=False)
        self.log("Shape of the {} df: {}".format(location, df.shape))
        self.log(msg)

    def fetch_alert_logs_data(self, sql): ## TODO: UNCOMMENT THIS BEFORE DEPLOYMENT
        connection = psycopg2.connect(
            host=self.ENDPOINT,
            port=self.PORT,
            user=self.USR,
            password=self.PASSWORD,
            database=self.DBNAME
        )
        self.log("Fetch Query: {}".format(sql))
        df = sqlio.read_sql_query(sql, connection)
        connection.close()
        self.log("Got data from the aurora alert logs: {}".format(df.shape))
        # df = pd.read_csv("D:/opt/ml/view_tnc_completeness.csv", sep='|')
        print("view df:",df.shape, df.columns)
        return df

    def clear_feedback_file(self, df, location, msg):
        self.log("In clear_feedback()")
        try:
            sql = self.table_parameters['pipelines'][self.PIPELINE]['sql_feedback']
            if sql != '': #self.PIPELINE == 'pipeline1' and
                df = self.fetch_alert_logs_data(sql)
                # check if the df contains any data beyond last 7 days:
                df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
                max_date = df['snapshot_date'].max()
                # set index from column Date
                df_temp = df.copy()
                df_temp = df_temp.set_index('snapshot_date')
                # if datetimeindex isn't order, order it
                df_temp = df_temp.sort_index()

                # last 7 days of date max_date
                df_temp = df_temp.loc[max_date - pd.Timedelta(days=3):max_date].reset_index()
                self.log(f"last 7 days data in feedback file, snapshot_dates: {str(df_temp['snapshot_date'].unique())}")
                ## check if impact is in the cols, if yes, then change it to severity
                df_temp = self.change_col(df_temp, 'impact', 'severity')
                for col in df_temp.columns:
                    if col.startswith('aa_'): self.change_col(df_temp, col, col[3:])
                self.save_file(df_temp, location, 'Saving last 7days data in S3')
        except Exception as e:
            self.log("error in clear_feedback(), {}".format(str(e)))
        return

    def change_col(self, df, exists, change_to):
        if exists in df.columns:
            df.rename(columns={exists: change_to}, inplace=True)
        return df

    def convert_to_numeric(self, df):
        for col in self.table_parameters['numeric_cols']:
            df[col] = pd.to_numeric(df[col])
        return df

    def convert_to_categoric(self, df):
        for col in self.table_parameters['pipelines'][self.PIPELINE]['categoric_cols']:
            df[col] = df[col].astype(str)
        return df

    def convert_to_date(self, df):
        self.log("in convert column to datetime")
        for col in self.table_parameters['date_cols']:
            self.log(col)
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        return df

    def filter_data(self, data, sign, conditions, col):
        print("in filter_data", sign, type(sign))
        print(data.shape)
        if sign == 'gtt':
            for key in conditions.keys():
                print('$$$$ in gtt', sign, key, conditions[key])
                try:
                    data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) > conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))
        if sign == 'gte':
            for key in conditions.keys():
                print('$$$$ in gte', sign, key, conditions[key])
                try:
                    data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) >= conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))
        if sign == 'ltt':
            for key in conditions.keys():
                try:
                    data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) < conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))
        if sign == 'lte':
            for key in conditions.keys():
                try:
                    data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) <= conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))
        if sign == 'nequalto':
            for key in conditions.keys():
                try:
                    if isinstance(conditions[key], (int, float)):
                        print("condition data is int ", conditions[key])
                        data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) != conditions[key]), 1, 0)
                    else:
                        print("condition data is not int or float ", conditions[key])
                        data[col] = np.where((data[col] == 1) & (data[key] != conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))
        if sign == 'equalto':
            for key in conditions.keys():
                try:
                    if isinstance(conditions[key], (int, float)):
                        print("condition data is int ", conditions[key])
                        data[col] = np.where((data[col] == 1) & (pd.to_numeric(data[key]) == conditions[key]), 1, 0)
                    else:
                        print("condition data is not int or float ", conditions[key])
                        data[col] = np.where((data[col] == 1) & (data[key] == conditions[key]), 1, 0)
                except Exception as e:
                    self.log("issue while applying filter condition: {} \n{}".format(col, e))

        return data

    def filter_alerts(self, conditions, data, col):
        signs = ['gtt', 'gte', 'ltt', 'lte', 'nequalto', 'equalto']
        for sign in signs:
            print("Sign: ", sign)
            if sign in conditions.keys():
                print("condition: ", conditions[sign])
                data = self.filter_data(data, sign, conditions[sign], col)
        return data

    def get_labels_completeness(self, data):
        """
        Function to label the data for completeness data which will be based on snapshot_date alone.
        Input:
            data: contains makes and model year, without model level data.
        Output:
            result_df: DataFrame that contains all the columns as in data along with calculated completeness and anomaly flags column
        """
        # print("In the TnC Completeness get_labels function")
        # print("Example input includes: ")
        # print(data.iloc[0:2, :])
        threshold = data[data[self.TARGET] > 0]['tnc_accepted_count'].sum() / data[data[self.TARGET] > 0][
            'cons_multi_channel_count'].sum() * 100
        threshold = round(threshold, 2)
        print(self.table_parameters['alert_conditions'])
        data['anomaly_flags'] = np.where((data[self.TARGET] < threshold - 10) | (
                data[self.TARGET] > threshold + 15)
                                         , 1, 0)
        data = self.filter_alerts(self.table_parameters['alert_conditions'], data, 'anomaly_flags')

        # data['anomaly_flags'] = np.where((data['cons_multi_channel_count'] > 0) &
        #                                  (data['device_count'] >= 500) & (
        #                                          (data[self.TARGET] < threshold - 10) | (
        #                                          data[self.TARGET] > threshold + 15))
        #                                  , 1, 0)

        data['change'] = round(data[self.TARGET] - threshold, 2)
        data['Threshold'] = threshold
        data = data.sort_values(self.TARGET)

        # print("get_labels function, example output includes: ")
        # print(data.iloc[0:2, :])
        # print("Leaving the get_labels function")
        return data, threshold

    def get_labels_completeness_all(self):
        result_df = pd.DataFrame()
        groups_df = self.data.groupby(['snapshot_date'])
        for grp in groups_df.groups.keys():
            group_df = groups_df.get_group(grp)
            group_df.reset_index(drop=True, inplace=True)
            group_df.fillna(value=0, inplace=True)
            group_df, self.tnc_completeness_threshold = self.get_labels_completeness(group_df)
            result_df = pd.concat([result_df, group_df])
        self.log("get_labels function, example output includes: ")
        self.log(result_df.iloc[0:2, :])
        self.log("Leaving the get_labels function")
        self.data = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def check_anomalies(self, group_df):
        if group_df[group_df['anomaly_flags']==1].shape[0]==0:
            # self.log("manually changing labels")
            group_df['anomaly_flags'] = np.where(group_df['change'] <=
                                                 -5 ## (-0.1 * self.table_parameters['pipelines'][self.PIPELINE]["minThreshold"])
                                                 , 1, 0)
            # print("$$$$ anomalies after check: ", group_df.shape, group_df[group_df['anomaly_flags']==1].shape)
        return group_df
    def get_labels(self):
        '''lables the train and inference data
        '''
        self.log('In labeling stage')
        print("df shape before labeling", self.data.shape)
        self.data.sort_values('snapshot_date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        groups_df = self.data.groupby(self.table_parameters['pipelines'][self.PIPELINE]['groupby'])
        result_df = pd.DataFrame()
        self.severity_thresholds = {}
        for grp in groups_df.groups.keys():
            # print(grp)
            group_df = groups_df.get_group(grp)
            group_df = group_df.sort_values('snapshot_date')
            group_df.reset_index(drop=True, inplace=True)
            group_df = self.get_change(group_df)
            # group_df['previous_'+self.TARGET] = group_df[self.TARGET].shift(1)
            # group_df['previous_'+self.TARGET].fillna(0, inplace=True)
            if self.inference_data_flag:
                group_df = group_df[group_df['snapshot_date'] == self.SNAPSHOT_DATE].reset_index(drop=True)

            else:
                y = abs(group_df['change'])
                if all([i == 0 for i in y]):
                    group_df["outliers"] = False
                else:

                    # print("Starting ThymeBoost labelling")
                    try:
                        boosted_model = TB.ThymeBoost()
                        output = boosted_model.detect_outliers(y,
                                                               trend_estimator='linear',
                                                               seasonal_estimator='fourier',
                                                               exogenous_estimator='ols',
                                                               seasonal_period=None,
                                                               global_cost='maicc',
                                                               fit_type='global')
                        group_df["outliers"] = output['outliers']
                    except:
                        group_df["outliers"] = False
                #
                # # print("Starting ThymeBoost labelling")
                # try:
                #     boosted_model = TB.ThymeBoost()
                #     output = boosted_model.detect_outliers(y,
                #                                            trend_estimator='linear',
                #                                            seasonal_estimator='fourier',
                #                                            exogenous_estimator='ols',
                #                                            seasonal_period=None,
                #                                            global_cost='maicc',
                #                                            fit_type='global')
                #
                #     group_df["outliers"] = output['outliers']
                # except:
                #     group_df["outliers"] = False
                # print("ThymeBoost labelling completed")
                group_df['anomaly_flags'] = group_df[['change', 'outliers']].apply(self.get_outliers, axis=1)
                group_df['anomaly_flags'] = pd.to_numeric(group_df['anomaly_flags'].replace([True, False], [1, 0]))
                group_df.drop('outliers', axis=1, inplace=True)
                group_df = self.check_anomalies(group_df) #check if thereare no anomalies in labelled data, then manually add using the min theshold
                # print("group_df before drop_na:", group_df.shape)
                group_df.dropna(inplace=True)
                # print("group_df after drop_na:", group_df.shape)
                group_df, group_severity_level = self.get_severity_level(group_df, threshold=
                    self.table_parameters['pipelines'][self.PIPELINE]["minThreshold"])
                self.severity_thresholds[str(grp)] = group_severity_level
                # print("Sample anomalies")
                # print(group_df[group_df['anomaly_flags'] == 1].head())

            result_df = pd.concat([result_df, group_df])

        self.data = result_df.reset_index(drop=True)
        print("results df shape:  ^^^^ ",self.data.shape)
        self.save_file(self.data,
                       str(os.path.join(self.feedbackDataPathOutput, self.filenames["feedback_trend"])),
                       "7-day 30-day trend")

        if not self.inference_data_flag:
            if self.PIPELINE == "pipeline1" and not self.feedback_data.empty:
                try:
                    # concat feedback data to train data
                    self.log("before combining with other data - self.data.shape: " + str(self.data.shape))
                    ## Combine with inference data:
                    print("############ ", self.saved_inference_df.shape, self.saved_inference_df)
                    if self.saved_inference_df.shape[0] > 0:
                        self.saved_inference_df = self.convert_to_categoric(self.saved_inference_df)
                        self.saved_inference_df = self.convert_to_date(self.saved_inference_df)
                        self.combine_with_labelled_data(self.saved_inference_df)
                    else:
                        self.log("self.saved_inference_df is empty")

                    if 'mdlyr' not in self.feedback_data.columns and 'model_year' in self.feedback_data.columns:
                        self.feedback_data.rename(
                            columns={'model_year': 'mdlyr'}, inplace=True)
                    else:
                        self.log("No model_year or mdlyr in the feedback data")
                    if self.feedback_data.shape[0] > 0:
                        self.feedback_data = self.convert_to_categoric(self.feedback_data)
                        self.feedback_data = self.convert_to_date(self.feedback_data)
                        self.combine_with_labelled_data(self.feedback_data)
                except Exception as e:
                    self.email_alert_developer("###### issue with feedback, not combining")
                    self.feedback_data = pd.DataFrame()

            severity_thresholds_df = pd.DataFrame(self.severity_thresholds).T
            # print(severity_thresholds_df)
            severity_thresholds_df = severity_thresholds_df[severity_thresholds_df['low'] != 100]
            severity_thresholds_mean = severity_thresholds_df.mean()
            self.severity_thresholds['mean'] = {'low': round(severity_thresholds_mean['low'], 2),
                                                'medium': round(severity_thresholds_mean['medium'], 2),
                                                'critical': round(severity_thresholds_mean['critical'], 2)}

            self.save_file(self.data, self.labelPath + '/' + self.filenames['labeled_data'],
                           'Saving train data with labels')
        self.log('Labeling done.')

    def combine_with_labelled_data(self, df):
        try:
            ## Remove all data >= snapshot_date
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            df = df.drop(df[df['snapshot_date'] >= self.SNAPSHOT_DATE].index)

            groupby_columns = self.table_parameters['pipelines'][self.PIPELINE]['groupby']
            groupby_columns.append('snapshot_date')
            self.log("df columns: " + str(df.columns))
            if 'mdlyr' not in df.columns and 'model_year' in df.columns:
                df.rename(columns={'model_year': 'mdlyr'}, inplace=True)
            else:
                self.log("No model_year or mdlyr in the feedback or inference data")
            severity_impact = ''
            if 'severity' in df.columns: severity_impact = 'severity'
            if 'impact' in df.columns: severity_impact = 'impact'

            if severity_impact == '': self.log(
                '*** ALERT!! No severity or impact columns in feedback or inference data')

            df['anomaly_flags'] = np.where(df[severity_impact].str.lower().isin(['low', 'medium', 'critical']), 1, 0)
            df['anomaly_flags'] = df['anomaly_flags'].map(str)
            self.data['anomaly_flags'] = self.data['anomaly_flags'].map(str)
            df['severity'] = np.where(df[severity_impact].str.lower().isin(['low', 'medium', 'critical']),
                                      df[severity_impact], 'No Anomaly')
            subset_columns_df = groupby_columns + ['severity', 'anomaly_flags']
            self.log("self.data columns: " + str(self.data.columns))
            self.log("self.data dtypes: " + str(self.data.dtypes))
            self.log("df dtypes: " + str(df.dtypes))

            # self.data = pd.concat([self.data, df[subset_columns_df]])
            # self.data.drop_duplicates(keep=('last'), inplace=True)
            self.data = self.data.merge(df[subset_columns_df],
                                        on=groupby_columns, how='left')
            # df = df.merge(self.data[subset_columns_df],
            #                 on=groupby_columns, how='left')
            self.data['anomaly_flags'] = self.data[['anomaly_flags_x', 'anomaly_flags_y']].apply(
                lambda x: x[0] if pd.isna(x[1]) else x[1], axis=1)
            self.data.drop(['anomaly_flags_x', 'anomaly_flags_y', 'severity_x', 'severity_y'], axis=1, inplace=True)
            self.data['anomaly_flags'] = self.data['anomaly_flags'].map(int)
            self.log("after combining self.data.shape: " + str(self.data.shape))
            self.data.drop_duplicates(subset=groupby_columns, keep="last")
            self.log("after drop duplicates self.data.shape: " + str(self.data.shape))
        except Exception as e:
            self.email_alert_developer("Issue while combining with labeled data, {}".format(str(e)))

    # def get_change(self, data):
    #     data['previous_' + self.table_parameters['date_cols'][0]] = data[self.table_parameters['date_cols'][0]].shift(1)
    #     for col in self.table_parameters['numeric_cols']:
    #         data['previous_' + col] = data[col].shift(1)
    #     data['previous_' + self.TARGET] = data[self.TARGET].shift(1)
    #     data['previous_' + self.TARGET].fillna(0, inplace=True)
    #     data['change'] = round(data[self.TARGET].diff().fillna(0), 2)
    #     return data

    def get_change(self, data):
        if self.shift_n == 1:
            self.prev_target_col = 'previous_' + self.TARGET
            self.current_target_col = self.TARGET
            data['previous_' + self.table_parameters['date_cols'][0]] = data[self.table_parameters['date_cols'][0]].shift(1)
            for col in self.table_parameters['numeric_cols']:
                data['previous_' + col] = data[col].shift(1)
            data[self.prev_target_col] = data[self.TARGET].shift(1)
            data[self.prev_target_col].fillna(0, inplace=True)
            data['change'] = round(data[self.TARGET].diff().fillna(0), 2)
        else:
            self.prev_target_col = str(self.shift_n) + 'to' + str(2*self.shift_n) + '_day_avg_' + self.TARGET
            self.current_target_col = str(self.shift_n) + '_day_avg_' + self.TARGET
            data[self.current_target_col] = round(data[self.TARGET].rolling(self.shift_n).mean(),2)
            data[str(self.shift_n)+'day_snapshot_dates'] = 0
            try:
                data[str(self.shift_n)+'day_snapshot_dates'][(self.shift_n-1):] = list(np.lib.stride_tricks.sliding_window_view(data.snapshot_date, self.shift_n))
            except:
                # print("issue with data:", data.shape)
                pass
            ## shift by shift_n day and then get rolling avg:
            data['shift_' + str(self.shift_n) + 'to' + str(2*self.shift_n) + '_day_' + self.TARGET] = \
                data[self.TARGET].shift(self.shift_n)
            data[self.prev_target_col] = \
                round(data['shift_' + str(self.shift_n) + 'to' + str(2*self.shift_n) + '_day_' + self.TARGET].rolling(7).mean(),2)
            data[str(self.shift_n) + 'to' + str(2*self.shift_n) + 'day_snapshot_dates'] = 0
            try:
                temp_dates = list(np.lib.stride_tricks.sliding_window_view(data.snapshot_date.shift(self.shift_n), self.shift_n))
                # print(temp_dates,"^^^ temp_dates")
                data[str(self.shift_n) + 'to' + str(2*self.shift_n) + 'day_snapshot_dates'][
                (self.shift_n - 1):] = temp_dates
            except Exception as e:
                # print(e,"issue with shift data:", data.shape)
                pass
            data['change'] = \
                round(data[self.current_target_col] \
            - data[self.prev_target_col],2)
        # data['30_day_avg_' + self.TARGET] = data[self.TARGET].rolling(30).mean()
        # data['shift_30to60_day_' + self.TARGET] = data[self.TARGET].shift(30)
        # data['30to60_day_avg' + self.TARGET] = data['shift_30to60_day_' + self.TARGET].rolling(30).mean()
        # data['change_30'] = data['30to60_day_avg' + self.TARGET] - data['30_day_avg_' + self.TARGET]
        return data

    def get_anomaly_flags(self):
        self.data['anomaly_flags'] = np.where(self.data['severity'].isin(['critical', 'medium', 'low']), 1, 0)
        #         If device_count column is in data, then if this value is <500, then dont consider the data point as anomaly
        # self.data.loc[self.data[self.table_parameters['denominator']] <= 500, ['anomaly_flags']] = 0
        self.data = self.filter_alerts(self.table_parameters['alert_conditions'], self.data, 'anomaly_flags')

    def get_severity_counts(self, anomalies_df):
        anomalies_df.loc[:, 'impact'] = anomalies_df.loc[:, 'severity']
        critical_count = len(anomalies_df[anomalies_df['impact'] == 'critical'])
        medium_count = len(anomalies_df[anomalies_df['impact'] == 'medium'])
        low_count = len(anomalies_df[anomalies_df['impact'] == 'low'])
        return critical_count, medium_count, low_count

    def get_outliers(self, x):
        change = abs(x[0])
        label = x[1]
        if change < 5:
            return False
        else:
            return label

    def assign_seviarity_value(self, x, low, med, critical):
        if x[1] == 1:
            if x[0] > critical:
                return 'critical'
            elif x[0] > med:
                return 'medium'
            elif x[0] >= low:
                return 'low'
            else:
                return "No Anomaly"
        else:
            return "No Anomaly"

    def assign_seviarity_value_inference(self, x, low, med, critical):
        thresholds = f"low: {low} - medium: {med} - critical: {critical}"
        if x[1] == 1:
            if x[0] > critical:
                print("CRITICAL! ", x[1])
                return 'critical', thresholds
            elif x[0] > med:
                return 'medium', thresholds
            elif x[0] >= low:
                return 'low', thresholds
            else:
                return "No Anomaly", thresholds
        else:
            return "No Anomaly", thresholds

    def get_severity_level(self, data, threshold=5):
        data_columns = list(data.columns)
        data['absolute_change'] = abs(data['change'])
        # Calculate severity level of an Anomaly point
        change_values = data['absolute_change']  # [data['absolute_change'] >= threshold]
        # low = threshold if change_values.min()<threshold else round(change_values.min(),2)
        if not change_values.empty:
            # print("===",change_values)
            low = round(np.percentile(change_values, self.table_parameters['pipelines'][self.PIPELINE]["minThreshold"]),
                        2)
            med = round(
                np.percentile(change_values, self.table_parameters['pipelines'][self.PIPELINE]["mediumThreshold"]), 2)
            critical = round(
                np.percentile(change_values, self.table_parameters['pipelines'][self.PIPELINE]["criticalThreshold"]), 2)
            # print(low, med, critical)
            data['severity'] = data[['absolute_change', 'anomaly_flags']].apply(self.assign_seviarity_value,
                                                                                args=(low, med, critical), axis=1)
        else:
            low = 100
            med = 100
            critical = 100
            data['severity'] = "No Anomaly"
        data['severity_thresholds'] = "{}".format({'low': low, 'medium': med, 'critical': critical})
        # print("{}".format({'low': low, 'medium': med, 'critical': critical}))
        return data[data_columns + ['severity','severity_thresholds']], {'low': low, 'medium': med, 'critical': critical}

    def email_alert_developer(self, message):
        try:
            email_subject = f'Developer Alert: {self.table_parameters["dashboard_name"]}-{" ".join(self.TRIAL_TYPE.split("_"))}'
            email_body = message
            self.log("Starting Developer Email alert")
            self.log(email_body)

            # sns_client = boto3.client('sns', region_name='us-east-1')
            # response = sns_client.publish(
            #     TopicArn=self.SnsRoleArnDeveloper,
            #     Message=email_body,
            #     Subject=email_subject,
            #     MessageStructure='string'
            # )
            # print(response)
        except Exception as e:
            print(e)
        self.log("Developer Email alert triggered")

    def process_data(self):
        self.log("In process_data function")
        #         print("Example of inputs:")
        #         print("Data:")
        #         print(self.data.iloc[0:2, :])
        #         print("\n")columns_to_preprocess
        self.log(self.data.columns)
        print('(in process data ) - *@Data : ' , self.data.shape)
        transformed_data = pd.DataFrame()
        if 'anomaly_flags' in self.data.columns: transformed_data['labels'] = self.data['anomaly_flags']
        if 'change' in self.data.columns: transformed_data['change'] = self.data['change']
        self.log(self.data.columns)
        transformed_data[self.TARGET] = self.data[self.TARGET]

        for col in self.table_parameters['date_cols']:
            self.data[col] = pd.to_datetime(self.data[col])
            transformed_data[col + '_year'] = self.data[col].dt.year
            transformed_data[col + '_month'] = self.data[col].dt.month
            transformed_data[col + '_day'] = self.data[col].dt.day
            transformed_data[col + '_DayOfTheWeek'] = self.data[col].dt.dayofweek
            transformed_data[col + '_WeekDay'] = (transformed_data[col + '_DayOfTheWeek'] < 5).astype(int)

        for col in self.table_parameters['numeric_cols']:
            transformed_data['norm_' + col] = (self.data[col] - self.data[col].min()) / (
                    self.data[col].max() - self.data[col].min())

        transformed_data = pd.concat([transformed_data, pd.get_dummies(
            self.data[self.table_parameters['pipelines'][self.PIPELINE]['categoric_cols']].T.agg('_'.join))],
                                     axis='columns')

        df_columns = list(transformed_data.columns)
        if 'labels' in df_columns:
            df_columns.remove('labels')
            transformed_data = transformed_data[df_columns + ['labels']]
        transformed_data.columns = transformed_data.columns.str.replace('[#,@,&,<,]', '')

        self.log("Data processed")
        #         print("Example output transformed_data:")
        #         print(transformed_data.iloc[0:2, :])
        #         print("\n")
        #         print("Leaving process_data function")

        return transformed_data

    def save_job_log_aurora(self, run_status, cpu_time): ## TODO
        try:
            # connection = psycopg2.connect(
            #     host=self.ENDPOINT,
            #     port=self.PORT,
            #     user=self.USR,
            #     password=self.PASSWORD,
            #     database=self.DBNAME
            # )

            # cursor = connection.cursor()
            # sql = self.config['qe_job_logs_update_query'].format(run_status=run_status, job_run_id=self.JOB_RUN_ID,
            #                                                      end_time=self.END_TIME, cpu_time=cpu_time,
            #                                                      user=self.USR)
            # cursor.execute(sql)
            # connection.commit()
            # connection.close()
            self.log("Job log update completed")
        except Exception as e:
            self.log(f"Issue while updating Job log: {e}")

    def feedback_identifier(self, x):
        prediction = x[0]
        feedback = x[1]
        if feedback == True:
            return prediction
        else:
            return abs(prediction - 1)

    def read_feedback_data(self, file_name):
        # Opening JSON file
        f = open(file_name)
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        f.close()
        feedback = pd.DataFrame(data['humanAnswers'][0]['answerContent']).T
        feedback.reset_index(inplace=True)
        feedback.rename(columns={'index': 'row'}, inplace=True)
        feedback['row'] = pd.to_numeric(feedback['row'])
        feedback.drop('disagree', axis=1, inplace=True)

        df = pd.DataFrame(data['inputContent']['Pairs'])
        feedback_df = pd.merge(df, feedback, on='row', how='left')
        feedback_df["true_label"] = feedback_df[['predictions', 'agree']].apply(self.feedback_identifier, axis=1)
        return feedback_df

    def read_input_file(self, input_file_path):
        df = pd.DataFrame()
        try:
            df = pd.read_csv(input_file_path)
            self.log("Shape of the data in file {} is {}".format(input_file_path, df.shape))
            if df.shape[0] == 0:
                self.log("No data in file {}".format(input_file_path))
        except Exception as e:
            self.log("Issue while reading data at {} \n{}".format(input_file_path, e))
        return df

    def read_inference(self):
        input_file_path = os.path.join(self.inferencePath, self.file_name)
        train_file_path = os.path.join(self.trainPath, self.train_file)
        print("combining inference and train files: {} {}".format(input_file_path,train_file_path) )
        self.log("Reading input data from{}".format(input_file_path))
        df_inference = self.read_input_file(input_file_path)
        df_train = self.read_input_file(train_file_path)
        df_inference['snapshot_date'] = pd.to_datetime(df_inference['snapshot_date'])
        df_train['snapshot_date'] = pd.to_datetime(df_train['snapshot_date'])
        dayTill = pd.to_datetime(df_inference['snapshot_date'].min())
        df = df_train[(df_train['snapshot_date'] < dayTill) &
                      (df_train['snapshot_date'] > dayTill - pd.Timedelta(days=round(int(self.shift_n)*2.5)))]
        self.log("Got {} days data from train:".format(round(int(self.shift_n)*2.5)))
        self.log("min max of df{}{}{}".format(df['snapshot_date'].min(), df['snapshot_date'].max(),df_inference['snapshot_date'].min()))
        self.data = pd.concat([df, df_inference])
        print("self.data min max", self.data['snapshot_date'].min(),self.data['snapshot_date'].max())
        del df, df_inference, df_train
        self.data.reset_index(drop=True, inplace=True)
        self.log(self.data.columns)
        self.data.dropna(inplace=True)
        self.data[self.TARGET] = self.data[self.TARGET].round(2)
        self.log("Combined inference with train, shape: {}\nmin:{}\nmax:{}".format(
                        self.data.shape,self.data['snapshot_date'].min(),self.data['snapshot_date'].max()))
    def get_shift_n_value(self):
        try:
            shift_n = self.table_parameters['pipelines'][self.PIPELINE]['shift_n'] ## make sure you are not overwriting self.shift_n that was set in __init__()
            self.shift_n = shift_n
            print("in shift_n func", self.shift_n)
        except Exception as e:
            self.log(f"issue with shift_n {e}")
            # self.shift_n = 1
        print("end of shift_n func", self.shift_n)
    def get_pipeline(self): ## Temp func to set pipeline parameter, TODO remove this once lambda is corrected
        print("in get_pipeline()", self.table_parameters['pipelines'].keys())
        pipeline = self.PIPELINE
        try:
            if self.shift_n == 7 and "pipeline3" in self.table_parameters['pipelines'].keys(): pipeline = "pipeline3"
            if self.shift_n == 30 and "pipeline4" in self.table_parameters['pipelines'].keys(): pipeline = "pipeline4"
            self.PIPELINE = pipeline
        except Exception as e:
            self.log(f"Issue while getting pipeline: '{e}'")
        finally:
            self.log(f"Finally pipeline is {self.PIPELINE} ")

    def main(self):
        # if 'model_yr' in self.data.columns: self.data.rename(columns={'model_yr': 'mdlyr'}, inplace=True)
        # if 'model' in self.data.columns: self.data.rename(columns={'model': 'mdl'}, inplace=True)

        try: ## TODO: Remove this before moving to dev
            self.create_filenames()
            self.get_local_file_paths()
            # self.get_pipeline()
            # self.get_shift_n_value()
            if self.inference_data_flag:
                self.read_inference()
                # input_file_path = os.path.join(self.inferencePath, self.file_name)
            else:
                input_file_path = os.path.join(self.inputPath, self.file_name)
                self.log("Reading input data from{}".format(input_file_path))
                self.data = self.read_input_file(input_file_path)
                self.log(self.data.columns)
                self.data.dropna(inplace=True)
                self.data[self.TARGET] = self.data[self.TARGET].round(2)

            if not self.inference_data_flag and self.PIPELINE == "pipeline1":
                actual_columns = list(self.data.columns)
                # actual_columns.append("true_label")
                # Read old feedback file
                self.log("Read previously combined feedback file")
                history_file_loc, feedback_file_loc = '', ''
                feedback_history_data = pd.DataFrame()
                self.log(f"List of files in the dir {self.feedbackDataPath}: {os.listdir(self.feedbackDataPath)}, looking for {self.filenames['feedback_file']}")
                for File in os.listdir(self.feedbackDataPath):
                    self.log(File)
                    if File == self.filenames['feedback_file'] : #.endswith("feedback.csv"):
                        feedback_file_loc = os.path.join(self.feedbackDataPath, File)
                        self.log("Feedback data present" + self.feedbackDataPath + '/' + File)
                        self.feedback_data = self.read_input_file(self.feedbackDataPath + '/' + File)
                        self.log("feedback_data {}".format(self.feedback_data.dtypes))
                    if File == self.filenames['feedback_history_file']:  #.endswith("feedback-history.csv"):
                        history_file_loc = os.path.join(self.feedbackDataPath, File)
                        self.log("Historic Feedback data present" + self.feedbackDataPath + '/' + File)
                        feedback_history_data = self.read_input_file(self.feedbackDataPath + '/' + File)
                        self.log("feedback_history_data" + str(feedback_history_data.dtypes))

                ## Combine latest feedback file with feedback history file:
                if self.feedback_data.shape[0] > 0 or feedback_history_data.shape[0] > 0:
                    self.feedback_data = pd.concat([feedback_history_data, self.feedback_data])
                    self.feedback_data.drop_duplicates(keep='last', inplace=True)

                    ## Save file at the historic location and clear the latest feedback file:
                    self.log("saving history {} {}".format(str(np.where(history_file_loc == '',
                                                                        os.path.join(self.feedbackDataPath,
                                                                                     self.filenames['feedback_history_file']),
                                                                        history_file_loc)),
                                                           self.feedbackDataPathOutput))
                    self.save_file(self.feedback_data,
                                   str(os.path.join(self.feedbackDataPathOutput,
                                                    self.filenames['feedback_history_file'])),
                                   "Saving feedback histroy")
                    self.log("saved history")
                    # self.clear_feedback_file(self.feedback_data, str(os.path.join(self.feedbackDataPathOutput, self.filenames['feedback_file'])), "Clearing feedback every 7th day")
                    # self.save_file(pd.DataFrame(),
                    #                str(os.path.join(self.feedbackDataPathOutput, self.filenames['feedback_file'])),
                    #                "Clearing feedback file")
                else:
                    self.log("Feedback files are empty")
                ## Getting saved inference to be combined with Train data:
                self.log(os.listdir(self.inferenceDataPathOutput))
                inference_file_loc = ''

                for File in os.listdir(self.inferenceDataPathOutput):
                    self.log(File)
                    if File.endswith("-inference.csv"):
                        inference_file_loc = os.path.join(self.inferenceDataPathOutput, File)
                        self.log("Inference data present" + str(inference_file_loc))
                        inference_data = self.read_input_file(inference_file_loc)
                        self.log("inference_data" + str(inference_data.dtypes))

            logFileName = "log_" + self.datetime_filename + ".txt"
            handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", self.logPath + "/" + logFileName))
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            root = logging.getLogger()
            root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
            root.addHandler(handler)
            # prepare_data_columns
            self.data = self.convert_to_numeric(self.data)
            self.data = self.convert_to_categoric(self.data)
            self.data = self.convert_to_date(self.data)
            if self.TABLE_NAME == "edo_edm.aggr_fact_tnc_metrics_completeness" and self.PIPELINE == "pipeline_snapshot":
                print('^^^ TnC Completeness, pipeline_snapshot (groupby snapshot_date) being run')
                self.get_labels_completeness_all()
            else:
                # Calculate change in value and label the data
                self.get_labels()

        except Exception as e:
            self.log("Pipeline run failed with error, {}".format(str(e)))
            print("Update job log in Aurora table")
            self.END_TIME = datetime.datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
            cpu_time = int((datetime.datetime.strptime(self.END_TIME, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                self.job_start_time, '%Y-%m-%d %H:%M:%S')).total_seconds())
            self.save_job_log_aurora("DATA PREPARATION FAILED", cpu_time)
            self.log("Job log update to Aurora table is completed")
            self.email_alert_developer("Pipeline run failed during data preparation with error, {}".format(str(e)))
            exit("Pipeline run failed during data preparation with error, {}".format(str(e)))


class AnomalyDetectionPipelineModelTraining(AnomalyDetectionPipeline):
    """  Main code for preprocessing, training and inference of data  """

    def __init__(self, path, shift_n, input_dict, debugMode):
        AnomalyDetectionPipeline.__init__(self, path, path, shift_n, input_dict, debugMode, inference_data_flag=False)
        self.log("AnomalyDetectionPipelineModelTraining __init__() complete")
        self.log("************Model training has Started************")

        self.train()

    def get_features(self, split_ratio=0.2, over_sampling=False):
        self.log("In get_features")
        train_features_file_path = os.path.join(self.trainPath, self.filenames['train_features'])
        train_labels_file_path = os.path.join(self.trainPath, self.filenames['train_labels'])
        test_features_file_path = os.path.join(self.testPath, self.filenames['test_features'])
        test_labels_file_path = os.path.join(self.testPath, self.filenames['test_labels'])
        processed_data_file_path = os.path.join(self.processedPath, self.filenames['processed_data'])
        # Stage 3: Process the input data
        self.log("Beginning processing of the labeled data")
        processed_data = self.process_data()
        self.log("Remove column with more than 99.5% zeros")
        self.log("Shape Before:" + str(processed_data.shape))
        column_cut_off = int(99 / 100 * len(processed_data))
        self.log("1")
        b = (processed_data == 0).sum(axis='rows')
        self.log("2")
        columns_to_consider = list(b[b <= column_cut_off].index.values)
        del b
        self.log("3")
        columns_to_consider.append('labels') if 'labels' not in columns_to_consider else columns_to_consider
        self.log("4")
        processed_data = processed_data[columns_to_consider]
        self.log("Shape After:" + str(processed_data.shape))
        self.log("Data has been processed")

        # Stage 5: Depending on inference, split input data into train and test and save all datafiles
        self.log("Splitting the processed data into train and test")

        # if processed_data.shape[0]!=0:
        features_labels = dict(
            zip(['training_features', 'test_features', 'training_labels', 'test_labels'], list(train_test_split(
                processed_data.iloc[:, :-1],
                processed_data.iloc[:, -1],
                test_size=split_ratio,
                random_state=10
            ))))
        # else:

        print('*@feature labels done')
        if over_sampling:
            resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='all'), random_state=42)
            features_labels['training_features'], features_labels['training_labels'] = resample.fit_resample(
                features_labels['training_features'], features_labels['training_labels'])
        #         print('features_labels',features_labels)
        self.save_file(features_labels['training_features'], train_features_file_path,
                       "Saving training features data to {}".format(train_features_file_path))
        self.save_file(features_labels['training_labels'], train_labels_file_path,
                       "Saving training labels data to {}".format(train_labels_file_path))
        self.save_file(features_labels['test_features'], test_features_file_path,
                       "Saving test features data to {}".format(test_features_file_path))
        self.save_file(features_labels['test_labels'], test_labels_file_path,
                       "Saving test labels data to {}".format(test_labels_file_path))
        self.save_file(processed_data, processed_data_file_path,
                       "Saving processed data to {}".format(processed_data_file_path))

        self.log("Processed data, train data, and test data saved")

        self.log("Processing stage completed")

        print('*@logs done')
        self.train_save_evaluate(list(processed_data.columns), features_labels)

    # ----------------------------------------------------------------------------------------------------------------------------

    def train_save_evaluate(self, processed_data_cols, features_labels):
        # Training
        self.log("In train_save_evaluate stage")

        # Stage 1: Train the ML model
        model = self.train_model(features_labels['training_features'], features_labels['training_labels'])

        # Stage 2: Save the ML model
        model_output_path = os.path.join(self.modelPath,
                                         self.filenames['model_file'])  ## TODO: why double join? check the function
        self.save_model(model, model_output_path)

        # Stage 1: Evaluate the model and save the evaluation dictionary

        evaluation_output_path = os.path.join(self.evalPath, self.filenames['evaluation'])
        #         self.evaluate_save(model,features_labels, evaluation_output_path)

        # Evaluation

        self.log("Evaluating the model")
        ## confusion: why are we saving same thing 2ce and are processed file path and train file path different?
        confusion_matrix, prediction_threshold = self.evaluate_model(model, features_labels)
        evaluation = {
            'Confusion Matrix': confusion_matrix,
            'Columns': list(processed_data_cols),
            'Train file Location': os.path.join(self.processedPath, self.filenames['processed_data'])
        }

        self.save_evaluation(evaluation, evaluation_output_path)
        self.log("Evaluation stage completed")

        self.log("Model metadata")
        model_details = {
            'Confusion Matrix': confusion_matrix,
            'Columns': list(processed_data_cols),
            'Prediction Threshold': float(prediction_threshold),
            'Train file Location': self.TRAIN_FILE_PATH
        }
        self.log("Model details type: " + str(type(model_details)))

        if not (self.TABLE_NAME == "edo_edm.aggr_fact_tnc_metrics_completeness" and self.PIPELINE == "pipeline_snapshot"):
            print('^^^ not TnC Completeness, pipeline_snapshot (groupby snapshot_date) being run')
            model_details['severity_thresholds'] = self.severity_thresholds
        self.model_metadata_save(model_details)
        self.email_alert_developer(json.dumps(model_details['Confusion Matrix'])) # TODO: Remove this comment

    def model_metadata_save(self, model_details):
        self.log("Saving the model metadata JSON")
        model_details_file_path = os.path.join(self.modelMetadataPath, self.filenames['model_details_file'])
        self.log("Saving the model metadata JSON to {}".format(model_details_file_path))
        with open(model_details_file_path, "w") as f:
            f.write(json.dumps(model_details))
        self.log("Model metadata json saved")

    def save_evaluation(self, evaluation, evaluation_output_path):
        self.log("Saving classification report to {}".format(evaluation_output_path))
        self.log("Saving the evaluation JSON")
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(evaluation))
        self.log("Evaluation json saved")

    def train_model(self, training_features, training_labels):
        self.log("Training the model")
        model = XGBClassifier(colsample_bytree=0.6000000000000001,
                              eta=0.4,
                              gamma=0.65,
                              learning_rate=0.015568964499909526,
                              max_depth=7,
                              min_sum_hessian_in_leaf=2.8410708495617514,
                              n_estimators=164,
                              seed=966096,
                              subsample=0.6000000000000001)
        model.fit(training_features, training_labels)
        self.model_algo = type(model).__name__
        return model

    def save_model(self, model, filename):
        """
        Function to save the machine learning model to a serialized format.

        Input:
            model: the machine learning model
        """
        self.log("Model has been trained, now saving the model")
        self.log(filename)
        model_output_directory = os.path.join("/opt/ml/processing/model", filename)  # TODO: remove this hardcoding
        self.log("Saving model to {}".format(model_output_directory))
        model_file = open(model_output_directory, 'wb')
        pickle.dump(model, model_file)
        model_file.close()
        self.log("Model saved")

    def evaluate_model(self, model, features_labels):
        """
        Function to evaluate the model's performance.

        Inputs:
            model: machine learning model
            test_features: table of features on which to test the model
            test_labels: labels against which to check the results of the model

        Output:
            evaluation: a dictionary showing the results of the model
        """
        self.log("In evaluation stage")

        # pred_labels = model.predict(test_features)
        prob_preds = model.predict_proba(features_labels['test_features'])
        prob_preds = [row[1] for row in prob_preds]
        fpr, tpr, thresholds = roc_curve(features_labels['test_labels'], prob_preds)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        threshold = thresholds[ix]
        pred_labels = [1 if x > threshold else 0 for x in prob_preds]
        return self.get_confusion_matrix(features_labels['test_labels'], pred_labels), threshold

    def get_confusion_matrix(self, test_labels, pred_labels):
        evaluation = {}
        try:
            tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
        except:
            tn, fp, fn, tp = 0, 0, 0, 0
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            print("No Anomalies in Train data")
            self.email_alert_developer('No Anomalies in Train data')

        evaluation['model_name'] = self.model_algo

        evaluation['accuracy'] = float(accuracy)
        evaluation['precision'] = float(precision)
        evaluation['recall'] = float(recall)
        evaluation['F1 score'] = float(f1)

        evaluation['True Positives'] = float(tp)
        evaluation['False Positives'] = float(fp)
        evaluation['False Negatives'] = float(fn)
        evaluation['True Negatives'] = float(tn)

        return evaluation

    def train(self):
        try:
            self.get_features(over_sampling=False)  # self.table_parameters['pipelines'][self.PIPELINE]['overSampling'])
        except Exception as e:
            self.log("Pipeline run failed with error, {}".format(str(e)))
            self.END_TIME = datetime.datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
            cpu_time = int((datetime.datetime.strptime(self.END_TIME, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                self.job_start_time, '%Y-%m-%d %H:%M:%S')).total_seconds())
            self.save_job_log_aurora("MODEL TRAINING FAILED", cpu_time)
            self.log("Job log update to Aurora table is completed")
            self.email_alert_developer("Pipeline run failed while training model with error, {}".format(str(e)))
            # exit("Pipeline run failed during data preparation with error, {}".format(str(e)))
            raise Exception("model training failed")


class AnomalyDetectionPipelineModelInference(AnomalyDetectionPipeline):
    """  Main code for preprocessing, training and inference of data  """

    def __init__(self, path, shift_n, input_dict, debugMode,  train_model_flag, train_file):
        AnomalyDetectionPipeline.__init__(self, path, train_file, shift_n, input_dict, debugMode, inference_data_flag=True)
        self.model = ''
        self.model_details = ''
        self.train_model_flag = train_model_flag
        self.totalAnomalies = 0
        self.log("AnomalyDetectionPipelineModelInference __init__() complete")
        self.log("************Inference stage has Started************")
        self.get_inference()
        ## Fetch last 7days data into feedback file
        self.clear_feedback_file(self.feedback_data,
                                 str(os.path.join(self.feedbackDataPathOutput, self.filenames['feedback_file'])),
                                 "Updating feedback every 7th day")

    def get_model(self):
        if self.train_model_flag == 'yes':
            ModelPath = self.modelPath
        else:
            ModelPath = self.pretrainedModelPath
        self.log("Check pretrained model folder {}".format(ModelPath))
        self.log(os.listdir(ModelPath))

        for File in os.listdir(ModelPath):
            self.log(File)
            if File == self.filenames['model_file']: # .endswith(".pkl"):
                self.log("model present" + ModelPath + '/' + File)
                self.log("Found a pretrained model")
                self.load_model(File, ModelPath)
                break

    def load_model(self, modelFile, ModelPath):
        self.model = pickle.load(open(ModelPath + '/' + modelFile, 'rb'))
        self.log(self.model)

    def inference(self, processed_data):
        """
        Function to make predictions on (processed) data, previously unseen by the model.

        Inputs:
            model: the machine learning model to be used
            data: the new data on which to make predictions
            end_data: the original input data, that was directly taken from the database

        Output:
            data: the data with an extra column called 'predictions'
        """

        # predictions = model.predict(data)
        self.log("processed_data.columns: " + str(processed_data.columns))
        prob_preds = self.model.predict_proba(processed_data)
        # print("prob_preds")
        # print(prob_preds)
        try:
            prob_preds = [row[1] for row in prob_preds]
        except:
            pass

        predictions = [1 if x > self.model_details['Prediction Threshold'] else 0 for x in prob_preds]

        processed_data['predictions'] = predictions
        self.save_file(processed_data, self.predictPath + '/' + self.filenames['processed_predicted_data'],
                       'Saving processed inference data with predictions')
        self.log("processed_data anomalies shape: " + str(processed_data[processed_data['predictions'] == 1].shape))
        self.log("processed_data shape: " + str(processed_data.shape))
        self.data['predictions'] = processed_data['predictions']
        # self.data.loc[self.data[self.table_parameters['denominator']] <= 500, ['predictions']] = 0
        # print('predictions in data: ', self.data['predictions'])
        self.data = self.filter_alerts(self.table_parameters['alert_conditions'], self.data, 'predictions')
        return processed_data

    def prepare_data(self):
        # get model metadata and match inference data columns with it.
        # get model_details file
        if self.train_model_flag == 'yes':
            modelDetailsPath = self.modelMetadataPath
        else:
            modelDetailsPath = self.pretrainedModelPath
        print("self.filenames['model_details_file']:",self.filenames['model_details_file'],modelDetailsPath, '$$$$$$$$$$$$')
        for File in os.listdir(modelDetailsPath):
            self.log(File)
            if File == self.filenames['model_details_file']: #.endswith(".json"):
                self.log("model details present at " + modelDetailsPath + '/' + File)
                self.log("Found model details")
                self.load_model_details(File, modelDetailsPath)
                break
        self.log('Model metadata {}'.format(self.model_details))
        self.log("\nBefore getting into process_data(), shape of the data: {}".format(self.data.shape))
        processed_data = self.process_data()
        self.log("Processed data cols: {}".format(processed_data.columns))
        self.log("Processed data shape: {}".format(processed_data.shape))
        input_data_cols = list(processed_data.columns)
        processed_data_cols = list(self.model_details['Columns'])
        processed_data_col_order = processed_data_cols[:-1]
        self.log('processed_data_col_order' + str(processed_data_col_order))
        for i in range(len(processed_data_cols)):
            col = processed_data_cols[i]
            if (col not in input_data_cols):
                processed_data[col] = np.zeros(processed_data.shape[0])
        processed_data = processed_data[processed_data_col_order]
        #         print('*****'*10,set(processed_data.columns)-set(self.model_details ['Columns']))
        #         print('*****'*10,set(self.model_details ['Columns']) - set(processed_data.columns))
        return processed_data

    def load_model_details(self, metadata_file, modelDetailsPath):
        metadata_file_path = os.path.join(modelDetailsPath, metadata_file)
        # JSON file
        f = open(metadata_file_path, "r")
        # Reading from file
        self.model_details = json.loads(f.read())
        f.close()

    def save_alert_log_aurora(self, alert_id, alerts, end_time, run_status='SUCCESS'):
        sql = self.config['alert_log_sql_insert']
        self.log("insert alert_log sql: " + sql)
        self.log("alert_log_sql_values: " + self.config['alert_log_sql_values'])
        first_run = True
        if self.TABLE_NAME == "edo_edm.aggr_fact_tnc_metrics_completeness" and self.PIPELINE == "pipeline_snapshot":
            print('^^^ TnC Completeness, pipeline_snapshot (groupby snapshot_date) being run')
            for alert in alerts:
                status_message = "Table: " + str(self.TABLE_NAME) + " - " + "Snapshot date: " + str(
                    self.SNAPSHOT_DATE) + " - " + alert
                self.log('status_message before making entry into Aurora alert logs: ' + str(status_message))
                if first_run:
                    sql_values = """(nextval('"edo_dms".seq_alert_log_id'),{alert_id},'{end_time}','{end_time}','{status}','{status_message}',0,0, CURRENT_DATE, '{added_user}', CURRENT_DATE,'{updt_user}','{trend_type}','{alert_severity}','{dashboard_link}','{severity_thresholds}')""".format(
                        alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                        added_user=self.USR, updt_user=self.USR, trend_type="NA", alert_severity="NA",
                        dashboard_link=self.config['email']['market_place_link'],
                        severity_thresholds=self.tnc_completeness_threshold)
                else:
                    sql_values = """, (nextval('"edo_dms".seq_alert_log_id'),{alert_id},'{end_time}','{end_time}','{status}','{status_message}',0,0, CURRENT_DATE, '{added_user}', CURRENT_DATE,'{updt_user}','{trend_type}','{alert_severity}','{dashboard_link}','{severity_thresholds}')""".format(
                        alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                        added_user=self.USR, updt_user=self.USR, trend_type="NA", alert_severity="NA",
                        dashboard_link=self.config['email']['market_place_link'],
                        severity_thresholds=self.tnc_completeness_threshold)
                sql += sql_values
                first_run = False
        else:
            if len(alerts['upward']) > 0:
                if self.shift_n == 1:
                    trend_type = 'upward'
                else:
                    trend_type = 'upward_' + str(self.shift_n)

                for alert in alerts['upward']:
                    alert_severity = next(iter(alert))
                    self.log("alert: " + str(alert))
                    self.log("severity_thresholds: " + str(alert['severity_thresholds']))
                    status_message = "Table: " + str(self.TABLE_NAME) + " - " + "Snapshot date: " + str(
                        self.SNAPSHOT_DATE) + " - " + \
                                     alert[alert_severity]
                    self.log('status_message before making entry into Aurora alert logs: ' + status_message)
                    if first_run:
                        sql_values = self.config['alert_log_sql_values'].format(
                            alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                            added_user=self.USR, updt_user=self.USR, trend_type=trend_type,
                            alert_severity=alert_severity
                            , current_date=datetime.date.today(),
                            dashboard_link=self.config['email']['market_place_link'],
                            severity_thresholds=alert['severity_thresholds'])

                    else:
                        sql_values = ',' + self.config['alert_log_sql_values'].format(
                            alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                            added_user=self.USR, updt_user=self.USR, trend_type=trend_type,
                            alert_severity=alert_severity
                            , current_date=datetime.date.today(),
                            dashboard_link=self.config['email']['market_place_link'],
                            severity_thresholds=alert['severity_thresholds'])
                    sql += sql_values
                    first_run = False
            if len(alerts['downward']) > 0:
                if self.shift_n == 1 : trend_type = 'downward'
                else: trend_type = 'downward_' + str(self.shift_n)
                for alert in alerts['downward']:
                    alert_severity = next(iter(alert))
                    self.log("alert: " + str(alert))
                    self.log("severity_thresholds: " + str(alert['severity_thresholds']))
                    status_message = "Table: " + str(self.TABLE_NAME) + " - " + "Snapshot date: " + str(
                        self.SNAPSHOT_DATE) + " - " + \
                                     alert[alert_severity]
                    if first_run:
                        sql_values = self.config['alert_log_sql_values'].format(
                            alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                            added_user=self.USR, updt_user=self.USR, trend_type=trend_type,
                            alert_severity=alert_severity
                            , current_date=datetime.date.today(),
                            dashboard_link=self.config['email']['market_place_link'],
                            severity_thresholds=alert['severity_thresholds'])
                    else:
                        sql_values = "," + self.config['alert_log_sql_values'].format(
                            alert_id=alert_id, end_time=end_time, status=run_status, status_message=status_message,
                            added_user=self.USR, updt_user=self.USR, trend_type=trend_type,
                            alert_severity=alert_severity
                            , current_date=datetime.date.today(),
                            dashboard_link=self.config['email']['market_place_link'],
                            severity_thresholds=alert['severity_thresholds'])
                    sql += sql_values
                    first_run = False
        self.log("SQL: " + sql)
        try:
            connection = psycopg2.connect(
                host=self.ENDPOINT,
                port=self.PORT,
                user=self.USR,
                password=self.PASSWORD,
                database=self.DBNAME
            )
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            connection.close()
            self.log("Alert log update completed")
        except Exception as e:
            self.log(f"Issue while updating alert log: {e}")

    def get_email_description(self, anomalies_df):
        critical_count, medium_count, low_count = self.get_severity_counts(anomalies_df)
        description = self.table_parameters['email_description'].format(no_of_anomalies=len(anomalies_df),
                                                                        entity=self.table_parameters["dashboard_name"])

        crosstab_df = pd.crosstab(anomalies_df['change_type'], anomalies_df['impact'])
        change_types = crosstab_df.index.values.tolist()

        if critical_count > 0:
            if "drop" in change_types:
                critical_drop_count = crosstab_df['critical']['drop']
            else:
                critical_drop_count = 0
            if "increase" in change_types:
                critical_increase_count = crosstab_df['critical']['increase']
            else:
                critical_increase_count = 0
            description = description + "Critical impact - {} (Upward - {}, Downward - {})\n".format(critical_count,
                                                                                                     critical_increase_count,
                                                                                                     critical_drop_count)
        if medium_count > 0:
            if "drop" in change_types:
                medium_drop_count = crosstab_df['medium']['drop']
            else:
                medium_drop_count = 0
            if "increase" in change_types:
                medium_increase_count = crosstab_df['medium']['increase']
            else:
                medium_increase_count = 0
            description = description + "Medium impact - {} (Upward - {}, Downward - {})\n".format(medium_count,
                                                                                                   medium_increase_count,
                                                                                                   medium_drop_count)
        if low_count > 0:
            if "drop" in change_types:
                low_drop_count = crosstab_df['low']['drop']
            else:
                low_drop_count = 0
            if "increase" in change_types:
                low_increase_count = crosstab_df['low']['increase']
            else:
                low_increase_count = 0
            description = description + "Low impact - {} (Upward - {}, Downward - {})".format(low_count,
                                                                                              low_increase_count,
                                                                                              low_drop_count)
        return description

    def inference_severity_calculation(self):
        self.data.loc[:, 'abs_change'] = self.data['change'].apply(lambda x: abs(x))
        for index, row in self.data.iterrows():
            grp = str(tuple(row[self.table_parameters['pipelines'][self.PIPELINE]['groupby']]))
            # print(self.model_details, "model details issue ^^^^^")
            try:
                severity_thresholds = self.model_details['severity_thresholds'][str(grp)]
                if severity_thresholds['low'] == 100:
                    severity_thresholds = self.model_details['severity_thresholds']['mean']
            except:
                severity_thresholds = self.model_details['severity_thresholds']['mean']
            # self.data['severity_thresholds'] = severity_thresholds
            severity_values, threshold_values = self.assign_seviarity_value_inference(
                row[['abs_change', 'predictions']],
                severity_thresholds['low'],
                severity_thresholds['medium'],
                severity_thresholds['critical'])
            # print("severity calculation")
            # print(severity_values)
            # print(threshold_values)
            self.data.at[index, 'severity'] = severity_values
            self.data.at[index, 'severity_thresholds'] = threshold_values
        self.log("after severity calculation")
        self.log(self.data.columns)
        self.save_file(self.data, self.resultPath + '/' + self.filenames['original_predicted_data'],
                       'Saving original inference data with predictions')
        # saving results file name of dependent pipeline in a json file in s3 to be used at the end of this ML file execution in its parent pipeline execution to send it to email module
        #
        # if self.PIPELINE != "pipeline1":
        processingJobResultFileName = {
            self.PIPELINE: {"resultFileName": self.filenames['original_predicted_data']}}
        with open(self.processingJobDetailsPath + '/' + self.filenames['processing_job_details'], 'w') as f:
            json.dump(processingJobResultFileName, f)
        self.save_inference(self.data, self.inferenceDataPathOutput + '/' + self.filenames['inference_save'],
                                'Saving original inference of Pipeline1 data with predictions in feedback folder')

    def email_alert(self):
        print("in email_alert")
        anomalies_df = self.data[self.data['predictions'] == 1].copy()
        # anomalies_df = self.data.head()
        self.totalAnomalies = len(anomalies_df)
        email_body = ''

        if self.totalAnomalies > 0:
            anomalies_df.reset_index(drop=True, inplace=True)

            # metric = self.table_parameters["metric"]
            # email_subject = self.table_parameters["email_subject"].format(trial_type=" ".join(self.TRIAL_TYPE.split('_')))

            def change_metrics(change):
                if change > 0:
                    change_type = "increase"
                elif change < 0:
                    change_type = "drop"
                else:
                    change_type = "no changes"
                return change_type

            def get_column_value_for_alert(column_name, row):
                if column_name in row:
                    column_value = row[column_name]
                else:
                    column_value = 0 # assuming column value to be numeric, returning 0 if the val is not present, e.g. in previous values of shift_n cases
                return column_value

            def get_numeric_column_values_for_alert(column_names, row):
                column_value, prev_column_value = '', ''
                for col in column_names:
                    try:
                        column_value += f' - {col}: {row[col]}'
                    except Exception as e:
                        print(f"issue with {col}{e}")
                        column_value += f' - {col}: 0' # returning 0 if the val is not present
                    previous_col = 'previous_' + col
                    try:
                        prev_column_value += f' - {previous_col}: {row[previous_col]}'
                    except Exception as e:
                        print(f"issue with {previous_col}{e}")
                        prev_column_value += f' - {previous_col}: 0' # returning 0 if the val is not present
                    column_value += prev_column_value
                # print("column_value: ",column_value)
                return column_value

            anomalies_df.loc[:, 'change_type'] = anomalies_df['change'].apply(lambda x: change_metrics(x))
            anomalies_df = anomalies_df[anomalies_df['change_type'] != "no changes"].reset_index(drop=True)
            anomalies_df = anomalies_df[anomalies_df['severity'] != "No Anomaly"].reset_index(drop=True)
            self.totalAnomalies = len(anomalies_df)
            if self.totalAnomalies > 0:
                description = self.get_email_description(anomalies_df)
                self.log(description)
                print("in email_alert()")
                # email_body += self.table_parameters["email_body"].format(
                #     entity=self.table_parameters["dashboard_name"]
                #     , trial_type=" ".join(self.TRIAL_TYPE.split('_'))
                #     , snapshot_date=self.SNAPSHOT_DATE
                #     , start_time=self.START_TIME
                #     , end_time=self.END_TIME
                #     , description=description
                # )
                anomalies_df.sort_values(['abs_change'], ascending=False, ignore_index=True, inplace=True)
                # if trial_type_or_metric_col in anomalies_df.columns:
                #     anomalies_df[trial_type_or_metric_col].str.title().replace('_', ' ', regex=True, inplace=True)
                email_body += """\n\n"""
                anomalies_df_increased = anomalies_df[anomalies_df['change_type'] == "increase"].reset_index(drop=True)
                anomalies_df_droped = anomalies_df[anomalies_df['change_type'] == "drop"].reset_index(drop=True)
                alerts = {'upward': [], 'downward': []}
                if len(anomalies_df_increased) > 0:
                    email_body += """\nUpward trend:\n"""
                    for index, row in anomalies_df_increased.iterrows():
                        alert = self.table_parameters['pipelines'][self.PIPELINE]["upward_trend_alert"].format(
                            trial_type=get_column_value_for_alert(column_name='trial_type', row=row),
                            data_metric=get_column_value_for_alert(column_name='metric', row=row),
                            metric_unit=get_column_value_for_alert(column_name='metric_unit', row=row),
                            device_group=get_column_value_for_alert(column_name='device_group', row=row),
                            device_type=get_column_value_for_alert(column_name='device_type', row=row),
                            validated_element=get_column_value_for_alert(column_name='validated_element', row=row),
                            make=get_column_value_for_alert(column_name='make', row=row),
                            model=get_column_value_for_alert(column_name='mdl', row=row),
                            mdlyr=get_column_value_for_alert(column_name='mdlyr', row=row),
                            rpo_cd=get_column_value_for_alert(column_name='rpo_cd', row=row),
                            country_cd=get_column_value_for_alert(column_name='country_cd', row=row),
                            src_sys_cd=get_column_value_for_alert(column_name='src_sys_cd', row=row),
                            src_sys=get_column_value_for_alert(column_name='src_sys', row=row),
                            royalty_ind=get_column_value_for_alert(column_name='royalty_ind', row=row),
                            event_dt=get_column_value_for_alert(column_name='event_dt', row=row),
                            type=get_column_value_for_alert(column_name='type', row=row),
                            validation_element=get_column_value_for_alert(column_name='validation_element', row=row),
                            addr_type=get_column_value_for_alert(column_name='addr_type', row=row),
                            src_nm=get_column_value_for_alert(column_name='src_nm', row=row),
                            batch_id=get_column_value_for_alert(column_name='batch_id', row=row),
                            change=round(get_column_value_for_alert(column_name='abs_change', row=row), 1),
                            previous=get_column_value_for_alert(column_name=self.prev_target_col, row=row),
                            current=get_column_value_for_alert(column_name=self.current_target_col, row=row),
                            metric=self.table_parameters["metric"],
                            impact=get_column_value_for_alert(column_name='impact', row=row).title(),
                            numeric_cols=get_numeric_column_values_for_alert(
                                column_names=self.table_parameters["numeric_cols"], row=row))
                        email_body += alert
                        alerts['upward'].append(
                            {row['impact']: alert, 'severity_thresholds': row['severity_thresholds']})
                if len(anomalies_df_droped) > 0:
                    email_body += """\nDownward trend:\n"""
                    for index, row in anomalies_df_droped.iterrows():
                        alert = self.table_parameters['pipelines'][self.PIPELINE]["downward_trend_alert"].format(
                            trial_type=get_column_value_for_alert(column_name='trial_type', row=row),
                            data_metric=get_column_value_for_alert(column_name='metric', row=row),
                            metric_unit=get_column_value_for_alert(column_name='metric_unit', row=row),
                            device_group=get_column_value_for_alert(column_name='device_group', row=row),
                            device_type=get_column_value_for_alert(column_name='device_type', row=row),
                            validated_element=get_column_value_for_alert(column_name='validated_element', row=row),
                            make=get_column_value_for_alert(column_name='make', row=row),
                            model=get_column_value_for_alert(column_name='mdl', row=row),
                            mdlyr=get_column_value_for_alert(column_name='mdlyr', row=row),
                            rpo_cd=get_column_value_for_alert(column_name='rpo_cd', row=row),
                            country_cd=get_column_value_for_alert(column_name='country_cd', row=row),
                            src_sys_cd=get_column_value_for_alert(column_name='src_sys_cd', row=row),
                            src_sys=get_column_value_for_alert(column_name='src_sys', row=row),
                            royalty_ind=get_column_value_for_alert(column_name='royalty_ind', row=row),
                            event_dt=get_column_value_for_alert(column_name='event_dt', row=row),
                            type=get_column_value_for_alert(column_name='type', row=row),
                            validation_element=get_column_value_for_alert(column_name='validation_element', row=row),
                            addr_type=get_column_value_for_alert(column_name='addr_type', row=row),
                            src_nm=get_column_value_for_alert(column_name='src_nm', row=row),
                            batch_id=get_column_value_for_alert(column_name='batch_id', row=row),
                            change=round(get_column_value_for_alert(column_name='abs_change', row=row), 1),
                            previous=get_column_value_for_alert(column_name=self.prev_target_col, row=row),
                            current=get_column_value_for_alert(column_name=self.current_target_col, row=row),
                            metric=self.table_parameters["metric"],
                            impact=get_column_value_for_alert(column_name='impact', row=row).title(),
                            numeric_cols=get_numeric_column_values_for_alert(
                                column_names=self.table_parameters["numeric_cols"], row=row))
                        email_body += alert
                        alerts['downward'].append(
                            {row['impact']: alert, 'severity_thresholds': row['severity_thresholds']})
                # email_body = email_body + """\n\nN.B.: Previous {metric} is the last day value.\n\n\nThis is an automated generated email. If you have any questions please send an e-mail to DMS team.""".format(
                #     metric=metric)

                # print("alert variable",self.table_parameters['pipelines'][self.PIPELINE]['alert'])
                # if self.table_parameters['pipelines'][self.PIPELINE]['alert']:
                #     print("Starting Email alert")
                #     print(email_body)
                #     sns_client = boto3.client('sns', region_name='us-east-1')
                #     response = sns_client.publish(
                #         TopicArn=self.SnsRoleARN,
                #         Message=email_body,
                #         Subject=email_subject,
                #         MessageStructure='string'
                #     )
                #     print(response)
                #     print("Email alert triggered")
                self.log("Save alert log to Aurora table")
                self.log(self.TRIAL_TYPE)
                alert_id = self.table_parameters['trial_type'][self.TRIAL_TYPE]['alert_id']
## TODO: uncomment save_alert_log
                self.save_alert_log_aurora(alert_id=alert_id, alerts=alerts, end_time=self.END_TIME,
                                           run_status='SUCCESS')

                self.log("Alert log saved to Aurora table")
            else:
                self.log("No anomalies detected in the data")
        else:
            self.log("No anomalies detected in the data")

        self.log(email_body)

    def email_alert_completeness(self):
        anomalies_df = self.data[self.data['predictions'] == 1].copy()
        self.totalAnomalies = len(anomalies_df)
        if self.totalAnomalies > 0:
            anomalies_df.reset_index(drop=True, inplace=True)
            metric = self.table_parameters["metric"]
            # email_subject = self.table_parameters["email_subject"].format(
            #     trial_type_metric=" ".join(self.TRIAL_TYPE.split('_')))
            description = "Total anomalies detected in {entity} data - {no_of_anomalies}".format(
                no_of_anomalies=self.totalAnomalies,
                entity=self.table_parameters["dashboard_name"])

            email_body = \
                """Control Entity: {entity}\nTrial Type:  {trial_type}\nData Snapshot Date: {snapshot_date}\nStart Time: {start_time}\nEnd Time: {end_time}\nDescription:  {description}\n""".format(
                    entity=self.table_parameters["dashboard_name"],
                    trial_type=" ".join(self.TRIAL_TYPE.split('_')),
                    start_time=self.START_TIME,
                    end_time=self.END_TIME,
                    description=description,
                    snapshot_date=self.SNAPSHOT_DATE)

            # anomalies_df.sort_values([TARGET], ascending=False, ignore_index=True, inplace=True)
            email_body += """\n\nAnomalies Detected:\n"""
            alerts = []
            for index, row in anomalies_df.iterrows():
                alert = """Type: {trial_type} - Make: {make} - Year: {mdlyr} has a {metric} of {current}%, whereas the threshold is {threshold}%.\n""".format(
                    trial_type=row['trial_type'],
                    make=row['make'],
                    mdlyr=row['mdlyr'],
                    current=row[self.TARGET],
                    # change=row['change_from_threshold'],
                    metric=metric,
                    threshold=self.tnc_completeness_threshold)
                email_body += alert
                alerts.append(alert)

            email_body = email_body + """\n\n\nThis is an automated generated email. If you have any questions please send an e-mail to DMS team."""
            self.log("Starting Email alert")
            self.log(email_body)
            # sns_client = boto3.client('sns', region_name='us-east-1')
            # response = sns_client.publish(
            #     TopicArn=self.SnsRoleARN,
            #     Message=email_body,
            #     Subject=email_subject,
            #     MessageStructure='string'
            # )
            # print(response)
            # print("Email alert triggered")
            self.log("Save alert log to Aurora table")
            self.log(self.TRIAL_TYPE)
            alert_id = self.table_parameters['trial_type'][self.TRIAL_TYPE]['alert_id']

            self.save_alert_log_aurora(alert_id=alert_id, alerts=alerts, end_time=self.END_TIME, run_status='SUCCESS')
            self.log("Alert log saved to Aurora table")
        else:
            self.log("No anomalies detected in the data")

    def get_inference(self):
        try:
            self.get_model()
            inference_file_present_flag = self.check_inference_datafram()
            self.log('inference check: {}'.format(inference_file_present_flag))

            if not inference_file_present_flag: raise Exception("No data for the snapshot date")

            self.inference(self.prepare_data())
            self.END_TIME = datetime.datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
            if self.TABLE_NAME == "edo_edm.aggr_fact_tnc_metrics_completeness" and self.PIPELINE == "pipeline_snapshot":
                print('^^^ TnC Completeness, pipeline_snapshot (groupby snapshot_date) being run')
                self.save_file(self.data, self.resultPath + '/' + self.filenames['original_predicted_data'],
                               'Saving original inference data with predictions')
                self.email_alert_completeness()
            else:
                self.inference_severity_calculation()
                self.email_alert()
            self.log("Update job log in Aurora table")
            cpu_time = int((datetime.datetime.strptime(self.END_TIME, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                self.job_start_time, '%Y-%m-%d %H:%M:%S')).total_seconds())
            self.save_job_log_aurora("SUCCESS", cpu_time)
            self.log("Job log update to Aurora table is completed")
        except Exception as e:
            self.log("Pipeline run failed with error, {}".format(str(e)))
            print("Update job log in Aurora table")
            self.END_TIME = datetime.datetime.now(est).strftime('%Y-%m-%d %H:%M:%S')
            cpu_time = int((datetime.datetime.strptime(self.END_TIME, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                self.job_start_time, '%Y-%m-%d %H:%M:%S')).total_seconds())
            self.save_job_log_aurora("INFERENCE FAILED", cpu_time)
            self.log("Job log update to Aurora table is completed")
            self.email_alert_developer("Pipeline run failed in inference stage with error, {}".format(str(e)))

    def get_saved_inference(self):
        inference_df = pd.DataFrame()
        Path = self.inferenceDataPathOutput
        self.log("Check if an inference file exists in folder {}".format(Path))
        self.log(os.listdir(Path))
        for File in os.listdir(Path):
            self.log("File found in inference path {} is {}".format(Path, File))
            if File.endswith(self.TARGET + "inference.csv"):
                self.log("inference file present" + Path + '/' + File)
                inference_df = self.get_csv(File, Path)
                break
        return inference_df

    def get_csv(self, file, path):
        input_file_path = os.path.join(path, file)
        df = pd.read_csv(input_file_path)  # pd.read_csv(path, nrows=100)
        self.log("Inference columns: {} \ninference shape: {}".format(df.columns, df.shape))
        return df

    def save_inference(self, df, location, msg):
        self.saved_inference_df = self.get_saved_inference()
        if self.saved_inference_df.shape[0] > 0:
            self.log("Inference file exists with shape {},\nadding latest inference file with shape {}".format(
                self.saved_inference_df.shape, df.shape))
            df = pd.concat([self.saved_inference_df, df])
            df = df.drop_duplicates(keep="last")

        df.to_csv(location, header=True, index=False)
        self.log("Shape of the {} df: {}".format(location, df.shape))
        self.log(msg)

    def check_inference_datafram(self):
        print(self.data.columns,'^^^^ chec for snapshot_date',self.SNAPSHOT_DATE, self.data.shape,
              self.data['snapshot_date'].unique())
        if self.data[self.data['snapshot_date'] == self.SNAPSHOT_DATE].shape[0] == 0:
            self.log("Inference dataframe is empty, checking redshift.")
            tmp = self.get_inference_data(
                self.table_parameters['pipelines'][self.PIPELINE]['sql_command_test'].format(
                    snapshot_date=self.SNAPSHOT_DATE,
                    trial_type=self.TRIAL_TYPE,
                    data_metric=self.DATA_METRIC))
            tmp[self.TARGET] = tmp[self.TARGET].round(2)
            tmp = self.convert_to_numeric(tmp)
            tmp = self.convert_to_categoric(tmp)
            tmp = self.convert_to_date(tmp)
            self.log("Fetched test data again: {}".format(tmp.shape))
            print(self.SNAPSHOT_DATE, type(self.SNAPSHOT_DATE))
            print(tmp.sort_values(by='snapshot_date', ascending=False).head(2))
            print(tmp.columns)
            self.log(str(tmp['snapshot_date'].unique()))
            for i in range(0, 8):
                if tmp[tmp['snapshot_date'] == self.SNAPSHOT_DATE].shape[0] == 0:
                    self.wait_sec(300) ## TODO: change to 300 secs
                    tmp = self.get_inference_data(
                        self.table_parameters['pipelines'][self.PIPELINE]['sql_command_test'].format(
                            snapshot_date=self.SNAPSHOT_DATE,
                            trial_type=self.TRIAL_TYPE,
                            data_metric=self.DATA_METRIC))
                    tmp[self.TARGET] = tmp[self.TARGET].round(2)
                    tmp = self.convert_to_numeric(tmp)
                    tmp = self.convert_to_categoric(tmp)
                    tmp = self.convert_to_date(tmp)
                    self.log("Fetched test data again {} time in pipeline: {}".format(i + 1, tmp.shape))

            if tmp[tmp['snapshot_date'] == self.SNAPSHOT_DATE].shape[0] == 0 : return False
            self.data = tmp
            self.log("saving new file at: {}".format(os.path.join(self.rawInferencePathOut, self.file_name)))
            self.save_file(self.data, os.path.join(self.rawInferencePathOut, self.file_name),
                           "Saving new inference file")
            # self.combine_inference_historic()
            self.data.dropna(inplace=True)

            if self.TABLE_NAME == "edo_edm.aggr_fact_tnc_metrics_completeness" and self.PIPELINE == "pipeline_snapshot":
                print('^^^ TnC Completeness, pipeline_snapshot (groupby snapshot_date) being run')
                self.get_labels_completeness_all()
            else:
                # Calculate change in value and label the data
                self.get_labels()

        return True

    def wait_sec(self, n):
        self.log(f'Inference dataframe is empty, adding {n} secs delay')
        i = 0
        for i in range(30, n + 30, 30):
            time.sleep(30)
        print(f'{i} secs wait')
        self.log('Wait time over reading the data again')
        return

    def get_redshift_conn(self):
        # print(f'''
        #     host={self.RedshiftHost},
        #     port={self.RedshiftPort},
        #     user={self.RedshiftUsername},
        #     password={self.RedshiftPassword},
        #     database={self.RedshiftDbname}''')
        conn = psycopg2.connect(
            host=self.RedshiftHost,
            port=self.RedshiftPort,
            user=self.RedshiftUsername,
            password=self.RedshiftPassword,
            database=self.RedshiftDbname
        )
        print("Redshift connection is established")
        return conn

    def get_inference_data(self, query):
        self.log("in get_inference_data")
        self.log(query)
        conn = self.get_redshift_conn()
        print('Redshift conn: ', conn)
        df = sqlio.read_sql_query(query, conn)
        conn.close()
        print("Redshift data read complete")
        df = df.drop_duplicates(keep="first")
        return df


def invoke_feedback_form(inferenceObj):
    try:
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lambda_payload = {"target": inferenceObj.TARGET,
                          "environment": inferenceObj.env,
                          "dashboard_name": inferenceObj.dashboard_name,
                          "dq_metric": inferenceObj.dq_metric,
                          "filename": inferenceObj.filenames['feedback_file'],
                          "pipeline":inferenceObj.PIPELINE,
                          "s3_path":'datalake/curated/mlmodels/anomalydetection/',
                          "feedback_file_path":f"{inferenceObj.dq_metric}/{inferenceObj.dashboard_name}/{inferenceObj.PIPELINE}/feedback_data/",
                          "feedback_page_path":f"{inferenceObj.dq_metric}/{inferenceObj.dashboard_name}/{inferenceObj.PIPELINE}/feedback_page/",
                          "pagename":'index.html',
                          }
        print("lambda_payload ", lambda_payload)
        lambda_client.invoke(FunctionName='sxm-edo-edm-dms-AnomalyDetection-FeedbackForm',
                             InvocationType='Event',
                             Payload=json.dumps(lambda_payload))

        print('Feedback Form initiated')
    except Exception as e:
        print("Issue while calling FeedbackForm lambda, {}".format(str(e)))
    return


