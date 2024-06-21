import boto3
from botocore.config import Config
import yaml
import datetime
from pytz import timezone

est = timezone('US/Eastern')
region = boto3.session.Session().region_name


def get_configs(bucket_name, file_key1, file_key2):
    # Initialize the S3 client
    s3_client = boto3.client('s3')

    def get_config(file_key):
        yaml_content = ''
        try:
            # Fetch the file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            # Read the file content
            content = response['Body'].read().decode('utf-8')

            # Parse the YAML content
            yaml_content = yaml.safe_load(content)

        except Exception as e:
            print(f"Error reading YAML file from S3: {str(e)}")

        return yaml_content

    config1 = get_config(file_key1)
    config2 = get_config(file_key2)
    config = {**config1, **config2}
    print("Config contents:", config)
    return config


def get_aws_info(config):
    aws_info = {'s3bucket': config['variables']['s3bucketName'],
                's3_path': config['code_file']['s3_path'],
                'ssm': boto3.client('ssm', region_name=region),
                'subnet_id': config['variables']['subnet_id'],
                'security_group_id': config['variables']['security_group_id'],
                'imageUri': config['variables']['imageUri']
                }
    return aws_info


def get_processing_inputs(aws_info, config):
    inference_file_path, inference_file_name = config['inference_file'].rsplit('/', 1)
    train_file_path, train_file_name = config['train_file'].rsplit('/', 1)
    processing_inputs = [
        {
            'InputName': config['code_file']['name'],
            'S3Input': {
                'S3Uri': 's3://{s3bucket}/{s3_path}/'.format(s3bucket=aws_info['s3bucket'],
                                                             s3_path=aws_info['s3_path']),
                'LocalPath': '/opt/ml/processing/input/code',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None'
            }
        },
        {
            'InputName': train_file_name,
            'S3Input': {
                'S3Uri': 's3://{s3bucket}/{s3_path}/'.format(s3bucket=aws_info['s3bucket'],
                                                             s3_path=train_file_path),
                'LocalPath': '/opt/ml/processing/input/train',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None'
            }
        },
        {
            'InputName': inference_file_name,
            'S3Input': {
                'S3Uri': 's3://{s3bucket}/{s3_path}/'.format(s3bucket=aws_info['s3bucket'],
                                                             s3_path=inference_file_path),
                'LocalPath': '/opt/ml/processing/input/inference',
                'S3DataType': 'S3Prefix',
                'S3InputMode': 'File',
                'S3DataDistributionType': 'FullyReplicated',
                'S3CompressionType': 'None'
            }
        },
    ]
    return processing_inputs


def get_processing_outputs(aws_info, config):
    inference_file_path, inference_file_name = config['inference_file'].rsplit('/', 1)
    results_file_path, results_file_name = config['results_file'].rsplit('/', 1)
    processing_outputs = {
        'Outputs': [
            {  # save logs
                'OutputName': 'processingJobDetails',
                'S3Output': {
                    'S3Uri': 's3://{s3bucket}/{s3_path}/logs/'.format(
                        s3bucket=aws_info['s3bucket'],
                        s3_path=aws_info['s3_path']),
                    'LocalPath': '/opt/ml/processing/processingJobDetails',
                    'S3UploadMode': 'Continuous'
                },

            },
            {
                'OutputName': 'model',
                'S3Output': {
                    'S3Uri': 's3://{s3bucket}/{s3_path}/model/'.format(
                        s3bucket=aws_info['s3bucket'],
                        s3_path=results_file_path),
                    'LocalPath': '/opt/ml/processing/model',
                    'S3UploadMode': 'Continuous'
                },

            },
            {  # save logs
                'OutputName': 'inference_results',
                'S3Output': {
                    'S3Uri': 's3://{s3bucket}/{s3_path}/results/'.format(
                        s3bucket=aws_info['s3bucket'],
                        s3_path=results_file_path),
                    'LocalPath': '/opt/ml/processing/output',
                    'S3UploadMode': 'Continuous'
                },

            }
        ]
    }

    return processing_outputs


def get_processing_resources(config, aws_info, domain, date_frequency):
    processing_resources = {
        'ClusterConfig': {
            'InstanceCount': 1,
            'InstanceType': config['to_train'][True]['instanceType'],
            'VolumeSizeInGB': config['to_train'][True]['diskSize']
        }
    }
    appSpec = {
        'ImageUri': aws_info['imageUri'],
        # 'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/{}'.format(
        #     config['code_file']['name'])],
        'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/{}'.format(
            config['code_file']['name'])],
        'ContainerArguments': ['--train-model', 'yes',  ## needs to be changed to train_model
                               '--train-filename', config['train_file'],
                               '--inference-filename', config['inference_file'],
                               '--result-file-path', config['results_file'],
                               '--processing-job-names', aws_info['processing_job_name'],
                               '--dataset-name', domain,
                               '--date-frequency', date_frequency
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


def set_datetime_filenames():
    current_date_time = datetime.datetime.now(est)
    current_year, current_month, current_day = str(current_date_time.year), str(current_date_time.month), str(
        current_date_time.day)
    current_hour, current_minute, current_second, current_microsec = str(current_date_time.hour), str(
        current_date_time.minute), str(
        current_date_time.second), str(current_date_time.microsecond)
    # datetime_filename_inference = "{}-{}-{}-{}-{}-{}-{}".format(current_year, current_month, current_day,
    #                                                             current_hour,
    #                                                             current_minute, current_second,
    #                                                             current_microsec)
    # datetime_filename = "/{}/{}/{}".format(current_year, current_month, current_day)
    processing_job_name_end = str(
        current_date_time.year % 100) + current_month + current_day + current_hour + current_minute + current_second + current_microsec
    return processing_job_name_end #datetime_filename_inference, datetime_filename,


def lambda_handler(event, context):
    print(event)
    # Define your S3 bucket name and the file key
    bucket_name = 'keynol-maverick'
    inference_file = event['Records'][0]['s3']['object']['key']
    domain = inference_file.split('/')[-1].rsplit('_',1)[0].lower().split("_")[0]
    date_frequency = inference_file.split('/')[-1].rsplit('_', 1)[0].lower().split("_")[1]
    report_type = inference_file.split('/')[-1].rsplit('_',1)[0].lower()
    metadata = 'MLcode/sagemaker/configs/metadata.yaml'
    domain_config = f'MLcode/sagemaker/configs/{domain}_config.yaml'
    config = get_configs(bucket_name, metadata, domain_config)
    config['inference_file'] = inference_file
    config['train_file'] = config['train_file'][report_type]
    config['results_file'] = config['results_file'][report_type]
    print("Config: ", config)
    aws_info = get_aws_info(config)
    print(aws_info)

    aws_info['processing_job_name'] = f"AnomalyDetection-{report_type.replace('_','-')}-{set_datetime_filenames()}"

    processing_inputs = get_processing_inputs(aws_info, config)
    processing_outputs = get_processing_outputs(aws_info, config)
    processing_resources, appSpec, NetworkConfig, Environment, tags = get_processing_resources(config, aws_info, domain, date_frequency)

    print("##############################")
    print(processing_inputs, processing_outputs, processing_resources, appSpec, NetworkConfig, Environment, tags)

    sagemaker_role_arn = "arn:aws:iam::850506728038:role/anomalydetection-sagemaker"  # get_iam_role_arn("anomalydetection-sagemaker")  ## Hardcoded Sagemaker role name!!
    print(sagemaker_role_arn)

    sagemaker_client = boto3.client("sagemaker", region_name=region,
                                    config=Config(connect_timeout=5, read_timeout=60,
                                                  retries={'max_attempts': 4})
                                    )

    print(f'''
        ProcessingJobName={aws_info['processing_job_name']},
        ProcessingInputs={processing_inputs},
        ProcessingOutputConfig={processing_outputs},
        ProcessingResources={processing_resources},
        AppSpecification={appSpec},
        RoleArn={sagemaker_role_arn},
        NetworkConfig={NetworkConfig},
        Environment={Environment},
        Tags={tags}'''
          )

    ## uncomment this when you want to trigger Sagemaker job:
    
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=aws_info['processing_job_name'],
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

    pr_job = sagemaker_client.describe_processing_job(ProcessingJobName=aws_info['processing_job_name'])
    processing_job_arn = pr_job['ProcessingJobArn']

    print("Processing Job: ", pr_job)
    print("Processing job ARN: ", processing_job_arn)

    return {
        'statusCode': 200,
        'body': "Success"
    }


