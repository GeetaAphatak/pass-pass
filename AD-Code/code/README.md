DQ Anomaly Detection
==============


## Project Overview:
This project involves building a generic and robust machine learning framework to predict anomalies in data within EDO data quality metrics. Pipeline contains following stages:

*	Triggering pipeline whenever data in quality metrics arrive – this is done via an SQL trigger on Aurora table, which invokes the pipeline whenever the Talend job log for data movement to Redshift is added to Aurora logs table.
*	Fetching data from Redshift DB – this is the first step of the pipeline where source data from Redshift DB is copied into S3 bucket. This is the train and inference data that is got from Redshift DB.
*	Labeling of the source datasets – involves marking the initial unlabeled dataset with "truth" values so that the machine learning algorithm can learn from it.
*	Feature extraction: involves changing the data that may have string columns and/or date columns to numeric so that they can be fed into the machine learning model
*	Model building: involves creating several supervised learning models to learn from the dataset and evaluate their performances
*	Getting inferences – Once the model is trained, the inference data is passed to it to find the anomalies.
*	Notifying Anomalies – Anomalies found in inference data are notified to business users.
*	Automation on the cloud in AWS – involves finding a mechanism to automatically return predicted anomalies when new data comes in. (This is ready now) Pipeline execution is initiated through an SQL trigger on AWS Aurora table

--------------
Steps to Deploy
--------------
Pre-requisites
* Aurora DB interface
* Terraform
* AWS CLI
* Git 

BitBucket link and command to run Terraform script
* Checkout the module by using git clone. This will download all the necessary files to the local server
git clone https://sxmbitbucket.corp.siriusxm.com/scm/edogmt/terraform_anomalydetection.git
* If running terraform script for first time then the modules and provider needs to be initialized first
  * cd terraform_anomalydetection
  * terraform init
* Run terraform plan to see the plan of the deployment. This is a dry run and if everything is validated then in the server output screen the outputs and summary would be displayed. On running this step, it might prompt for additional user input on standard input. Please refer to next section for configuring parameters
* Once the plan is executed and the output looks valid, to apply the deployment run terraform apply. Please note same as plan it would ask for additional user input. Additionally, it would ask for user confirmation to deploy the AWS components
* Configuring system and environment variables 
  * Manually change environment variable ‘environment’ in terraform.tfvars file and also in main.tf’s s3 bucket name.

Infrastructure
---------------
### Components 
| Name                                                            | Component                                                     | Resource Type                              | IAM Role(s)                                                        | Tags                                                                                                                                                                                       | Description                                                                                                                                                                                                                 |
| --------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Aurora Trigger                                                  | edm\_func\_ad\_lambdainvoke<br>edm\_trigger\_ad\_lambdainvoke | Database components - Function and Trigger | NA                                                                 | NA                                                                                                                                                                                         | To invoke lambda function from Aurora DB, trigger - edm\_trigger\_ad\_lambdainvoke, needs to be set on edo\_dms.qe\_query\_logs table on insert.<br>This trigger inturn calls the the function edm\_func\_ad\_lambdainvoke. |
| sxm-edo-edm-dms-AnomalyDetection-$(var.environment)             | Lambda                                                        | Python 3.7                                 | Appl-edo-edm-dms-anomalydetection-lambdarole-${var.environment}    | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) | This Lambda function is invoked from Aurora trigger, this inturn creates a Sagemaker processing job that runs ML module to train model and get inferences                                                                   |
| pandas\_psycopg2                                                | Lambda layer                                                  | Python 3.7                                 | Appl-edo-edm-dms-anomalydetection-lambdarole-${var.environment}    |                                                                                                                                                                                            | Contains pandas and psycopg2 packages required by above lambda function                                                                                                                                                     |
| cryptography                                                    | Lambda layer                                                  | Python 3.7                                 | Appl-edo-edm-dms-anomalydetection-lambdarole-${var.environment}    |                                                                                                                                                                                            | Contains cryptography package that is required by above Lambda function                                                                                                                                                     |
| sxm-edo-${var.environment}                                      | S3                                                            | Storage                                    | NA                                                                 | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) | Place data fetched from Redshift into S3 Bucket, also all the code and inference results file are saved here.                                                                                                               |
| sxm-edo-edm-anomalydetection-ml-pipeline-${var.date}            | Sagemaker processing job                                      | Python 3.6                                 | Appl-edo-edm-dms-anomalydetection-sagemakerrole-${var.environment} | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) | Sagemaker processing job to run the ML pipeline - train model and get inferences                                                                                                                                            |
| sxm-edo-edm-dms-AnomalyDetectionNotification-$(var.environment) | SNS                                                           | Email                                      | \_\_default\_policy\_ID                                            | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) | Anomalies are notified thru this module.                                                                                                                                                                                    |

### IAM Roles
| Name                                                               | Component | Resource Type | Policies                                                  | Tags                                                                                                                                                                                       | Description                                             |
| ------------------------------------------------------------------ | --------- | ------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| Appl-edo-edm-dms-anomalydetection-lambdarole-${var.environment}    | IAM Role  | Lambda        | sxm-edo-edm-${var.environment}-anomalydetection-lambda    | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) |                                                         |
| Appl-edo-edm-dms-anomalydetection-sagemakerrole-${var.environment} | IAM Role  | Lambda        | sxm-edo-edm-${var.environment}-anomalydetection-sagemaker | access-department : edm<br>access-team : dms<br>BillTagId : edo-edm-dms-SagemakerAD-$(var.environment)<br>access-org : edo<br>TagID : edo-edm-dms-anomalydetection-role-$(var.environment) |                                                         |
| rds-lambda-role                                                    | IAM Role  | RDS           | rds\_lambda\_policy                                       |                                                                                                                                                                                            | Role created by DBA for accessing Lambda from Aurora DB |

### IAM Policies
| Name                                                      | Component  | Resource Type    | Permissions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------- | ---------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| rds\_lambda\_policy                                       | IAM Policy | DBA managed      | lambda:InvokeFunction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| sxm-edo-edm-${var.environment}-anomalydetection-lambda    | IAM Policy | Customer managed | <br>"logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"<br>"s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"<br>“kms:Decrypt","kms:DescribeKey","kms:GenerateDataKey”<br>“sagemaker:CreateAction","sagemaker:CreateProcessingJob","sagemaker:DescribeProcessingJob","sagemaker:ListProcessingJobs","sagemaker:StopProcessingJob”, “sagemaker:CreateTrainingJob”<br>“redshift-data:ExecuteStatement","redshift-data:ListDatabases","redshift:GetClusterCredentials”<br>SNS:Publish<br>iam:PassRole                                                                                                                                                                                                                                                                                                                                                                               |
| sxm-edo-edm-${var.environment}-anomalydetection-sagemaker | IAM Policy | Customer managed | "logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"<br>"s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"<br>"sagemaker:CreateAction","sagemaker:CreateProcessingJob","sagemaker:DescribeProcessingJob","sagemaker:ListProcessingJobs","sagemaker:StopProcessingJob"<br>"kms:Decrypt","kms:DescribeKey","kms:GenerateDataKey"<br>"lambda:GetFunction","lambda:ListFunctions","lambda:InvokeFunction"<br>"iam:GetRole"<br>"ec2:CreateNetworkInterface",<br>"ec2:DescribeDhcpOptions",<br>"ec2:DescribeNetworkInterfaces",<br>"ec2:DeleteNetworkInterface",<br>"ec2:DescribeSubnets",<br>"ec2:DescribeSecurityGroups",<br>"ec2:DescribeVpcs",<br>"ec2:CreateNetworkInterfacePermission"<br>"sns:Publish"<br>"ecr:BatchGetImage","ecr:GetAuthorizationToken",<br>"ecr:BatchCheckLayerAvailability",<br>"ecr:CreateRepository",<br>"ecr:Describe\*",<br>"ecr:GetDownloadUrlForLayer" |
| \_\_default\_policy\_ID                                   | IAM Role   | SNS              | "SNS:Subscribe",<br>        "SNS:SetTopicAttributes",<br>        "SNS:RemovePermission",<br>        "SNS:Receive",<br>        "SNS:Publish",<br>        "SNS:ListSubscriptionsByTopic",<br>        "SNS:GetTopicAttributes",<br>        "SNS:DeleteTopic",<br>        "SNS:AddPermission"<br>sns:Publish                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

### VPC Endpoints
| Name                         | Component    | Resource Type    | VPC Endpoint | Tags | Description                                                        |
| ---------------------------- | ------------ | ---------------- | ------------ | ---- | ------------------------------------------------------------------ |
| Parameter Store VPC Endpoint | VPC Endpoint | Customer managed | NA           | NA   | Required to access Parameter Store values that contain credentials |
| Lambda VPC Endpoint          | VPC Endpoint | Customer managed | NA           | NA   | Required to invoke lambda function from Aurora                     |



SQL Components
------------------
There are 2 components a Trigger on qe_query_logs table on insert and Function on edo_dms schema of Aurora database.
These files can be found in RdsLambdaTrigger folder.

SNS Subscription
----------------
Manually create subscriptions through AWS console. Visit SNS Topic created via Terraform
1.	Visit https://console.aws.amazon.com/sns/v3/home?region=us-east-1#/topics
2.	Search for ‘sxm-edo-edm-dms-AnomalyDetectionNotification-${var.environment}’
3.	Open this link
4.	Under ‘Subscriptions’ → ‘Create Subscription’
5.	Select Protocol → Email
6.	Endpoint → Email id to be registered
7.	Click → ‘Create Subscription’ button

Getting KMS ARN
----------------
KMS ARN is required to access S3 bucket and save our files. 
To find KMS ARN: 

1.	Open any file in the s3 bucket 
2.	KMS ARN is present under Server-side encryption settings

Getting Subnet IDs & Security Groups
----------------
Subnet is required to have all services in the same network. 
To find Subnet ID: 

1.	Go to RDS 
2.	Open the Database instance that you connect to for Aurora
3.  Subnets are listed under Connectivity & Security tab
4.  Pick any 2 from the list and confirm there are enough 'Available IPv4 addresses' listed on the page when a subnet link is clicked.
5.  Get the security group listed on the RDS Database page.
6.  Another security group should be got from Redshift Cluster page.


Get Docker Image URI
----------------
A docker image is required to run our sagemaker processing jobs. This image is hosted in an ECR repository.

To get this URI, go to ECR → Repository and search for 'edo-edm-sagemaker-scikit-learn'
 

