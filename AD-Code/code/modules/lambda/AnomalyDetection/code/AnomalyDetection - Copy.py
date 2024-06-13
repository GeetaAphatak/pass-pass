"""
#Lambda Function that will initiate Sagemaker processing job that will inturn train a model and perform predictions.

"""

import os
import json
import boto3
from botocore.config import Config
import time
import datetime
from pytz import timezone
import yaml

est = timezone('US/Eastern')
region = boto3.session.Session().region_name


def get_config_details(sagemaker, s3bucket, file1, file2):
    def get_file_contents(sagemaker, s3bucket, file):
        yaml_localFilename = '/tmp/file.yaml'
        print("s3bucket, yaml_config_file_full_path, yaml_localFilename", s3bucket, file,
              yaml_localFilename)
        print("sagemaker['s3c']", sagemaker['s3c'])
        sagemaker['s3c'].download_file(s3bucket, file, yaml_localFilename)
        with open(yaml_localFilename) as f:
            config = yaml.safe_load(f)
        return config

    config1 = get_file_contents(sagemaker, s3bucket, file1)
    config2 = get_file_contents(sagemaker, s3bucket, file2)
    config = {**config1, **config2}
    print("Config contents:", config)
    return config



def get_aws_info(config):
    aws_info = {'s3bucket': config['s3bucketName'],
                's3_path': config['code_file']['s3_path'],
                'ssm': boto3.client('ssm', region_name=region),
                'subnet_id': config['subnet_id'].split(","),
                'security_group_id': config['security_group_id'].split(",")}
    return aws_info

def get_sagemaker_info():
    sagemaker = {'s3c': boto3.client("s3"),
                 's3r': boto3.resource("s3", region_name=region),
                 'redshift': boto3.client('redshift', region_name=region),
                 'client': boto3.client("sagemaker", region_name=region,
                                        config=Config(connect_timeout=5, read_timeout=60,
                                                      retries={'max_attempts': 20})
                                        )
                 }
    print(sagemaker['client'].meta.config.retries)
    return sagemaker


def get_processing_inputs(config):
    processing_inputs = [
        {
            'InputName': config['code_file']['name'],
            'S3Input': {
                'S3Uri': config['code_file']['s3_path'].format(s3bucket=config['s3bucketName'],
                                                               s3_path=config['code_file']['s3_path']),
                'LocalPath': '/opt/ml/processing/input/code',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None'
            }
        },
    ]
    return processing_inputs


def get_processing_outputs(config):
    processing_outputs = {
        'Outputs': [
            {  # save logs
                'OutputName': 'processingJobDetails',
                'S3Output': {
                    'S3Uri': 's3://keynol-maverick/MLcode/sagemaker/logs/processingJobDetails/',
                    'LocalPath': '/opt/ml/processing/processingJobDetails',
                    'S3UploadMode': 'Continuous'
                },

            },
        ]
    }

    return processing_outputs

def get_processing_resources(config, aws_info):
    processing_resources = {
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': config['to_train'][True]['instanceType'],
            'VolumeSizeInGB': config['to_train'][True]['diskSize']
        }
    }
    appSpec = {
        'ImageUri': config['imageUri'],
        # 'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/{}'.format(
        #     config['code_file']['name'])],
        'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/{}'.format(
            config['code_file']['name'])],
        'ContainerArguments': ['--train-model', 'yes',  ## needs to be changed to train_model
                               #   '--historic-ranges', str(event_json['historic_ranges']),
                               # '--metric-name', str(event_json['target']),
                               # '--table-name', str(event_json['table_name']),
                               # '--train-filename', filenames['train'],
                               # '--test-filename', filenames['test'],
                               # '--inference-filename', filenames['inference'],
                               # '--result-file-path', filenames['s3_bucket'] + filenames['results_path'],
                               # '--processing-job-names', filenames['processing_job_name']
                               ]
    }
    NetworkConfig = {
        "VpcConfig": {
            "Subnets": aws_info['subnet_id'],
            "SecurityGroupIds": aws_info['security_group_id']
        }
    }
    Environment = {}
    tags = [
        {
            "Key": "access-department",
            "Value": "ML"
        },
        {
            "Key": "access-team",
            "Value": "ML"
        },
        {
            "Key": "BillTagId",
            "Value": "AnomalyDetection"
        },
        {
            "Key": "access-org",
            "Value": "Keynol"
        },
        {
            "Key": "TagID",
            "Value": "AnomalyDetection-sagemaker"
        }
    ]
    return processing_resources, appSpec, NetworkConfig, Environment, tags


def get_iam_role_arn(role_name):
    # Create an IAM client
    iam_client = boto3.client('iam')

    try:
        # Get the role details
        response = iam_client.get_role(RoleName=role_name)

        # Extract and return the ARN
        role_arn = response['Role']['Arn']
        return role_arn

    except iam_client.exceptions.NoSuchEntityException:
        print(f"The role {role_name} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def lambda_handler(event_info, context):
    print(event_info)
    sagemaker = get_sagemaker_info()
    config = get_config_details(sagemaker, s3bucket='keynol-maverick',
                                file1='MLcode/sagemaker/walmart_config.yaml',
                                file2='MLcode/sagemaker/metdata.yaml')
    aws_info = get_aws_info(config)
    processing_inputs = get_processing_inputs(config)
    processing_outputs = get_processing_outputs(config)
    processing_resources, appSpec, NetworkConfig, Environment, tags = get_processing_resources(config, aws_info)
    sagemaker_role_arn = get_iam_role_arn("anomalydetection-sagemaker") ## Hardcoded Sagemaker role name!!

    processing_job_name = "AnomalyDetection_test"

    response = sagemaker['client'].create_processing_job(
        ProcessingJobName=processing_job_name,
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig=processing_outputs,
        ProcessingResources=processing_resources,
        AppSpecification=appSpec,
        RoleArn=sagemaker_role_arn,
        NetworkConfig=NetworkConfig,
        Environment=Environment,
        Tags=tags
    )
    print("Processing Job Started. \n", response)
    print("Please go to Sagemaker for any further error checking")

    pr_job = sagemaker['client'].describe_processing_job(ProcessingJobName=processing_job_name)
    processing_job_arn = pr_job['ProcessingJobArn']

    print("Processing Job: ", pr_job)
    print("Processing job ARN: ", processing_job_arn)

    return {'statusCode': 200, 'body': json.dumps('Lambda Function complete')}
