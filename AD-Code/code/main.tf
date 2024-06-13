
data "archive_file" "AnomalyDetection" {
  type ="zip"
  source_file = var.AnomalyDetectionLambdaSourceFile
  output_path = var.AnomalyDetectionLambdaZipPath
}

data "archive_file" "AnomalyDetectionResults" {
  type ="zip"
  source_file = var.AnomalyDetectionResultsLambdaSourceFile
  output_path = var.AnomalyDetectionResultsLambdaZipPath
}


data "aws_s3_bucket" "bucket" {
  bucket = local.s3bucketName
}

#  usage note:  run local first then uncomment below and do a terraform init to copy state to bucket
#
#This module requires terraform Version 0.12.0+ and uses S3 for tfstate

terraform {
  required_version = ">= 0.12.0"
  backend "s3" {
    region  = "us-east-1"
    bucket  = "keynol-maverick"
    key     = "MLcode/terraform/terraform.tfstate"
    encrypt = "true"
    acl     = "bucket-owner-full-control"
  }
}

# ---------------------------------------------------------------------------------------------------------------------
#create lambda function policy
# ---------------------------------------------------------------------------------------------------------------------
module "AnomalyDetectionLambdaPolicy" {
  source = "./modules/policy/AnomalyDetectionLambda"
  name = "anomalydetection-lambda"
  description = "Policy for AnomalyDetection Lambda function"
  s3bucket = local.s3bucketName
  tags = local.AnomalyDetectionRoleTags
}

# ---------------------------------------------------------------------------------------------------------------------
#create lambda function role
# ---------------------------------------------------------------------------------------------------------------------

module "AnomalyDetectionLambdaRole" {
  source = "./modules/role/AnomalyDetectionLambda"
  name = "anomalydetection-lambdarole"
  tags = local.AnomalyDetectionRoleTags
  AnomalyDetectionLambdaPolicyArn = module.AnomalyDetectionLambdaPolicy.Policy_arn
}

# ---------------------------------------------------------------------------------------------------------------------
#create lambda layer
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_lambda_layer_version" "pandas_lambda_layer" {
  filename   = "./modules/lambda/lambdaLayer/pandas-lambda-layer.zip"
  layer_name = "pandas_lambda_layer"
  compatible_runtimes = ["python3.10"]
}

resource "aws_lambda_layer_version" "RedshiftConn_layer" {
  filename   = "./modules/lambda/lambdaLayer/RedshiftConn.zip"
  layer_name = "RedshiftConn"
  compatible_runtimes = ["python3.10"]
}



# ---------------------------------------------------------------------------------------------------------------------
#create lambda function
# ---------------------------------------------------------------------------------------------------------------------

module "AnomalyDetectionLambdaFunction" {
  source = "./modules/lambda/AnomalyDetection"
  name = local.AnomalyDetectionLambdaName
  lambda_zip_path = var.AnomalyDetectionLambdaZipPath
  handler = var.AnomalyDetectionLambdaHandler
  runtime = var.AnomalyDetectionLambdaRuntime
  roleArn = module.AnomalyDetectionLambdaRole.Role_arn
  memory_size = var.AnomalyDetectionLambdaMemory
  timeout = var.AnomalyDetectionLambdaTimeout
  tags = local.AnomalyDetectionLambdaTags
  layers = [aws_lambda_layer_version.pandas_lambda_layer.arn,aws_lambda_layer_version.RedshiftConn_layer.arn]
  subnet_id = local.subnet_id
  security_group_id = local.security_group_id
}


module "AnomalyDetectionResultsLambdaFunction" {
  source = "./modules/lambda/AnomalyDetectionResults"
  name = local.AnomalyDetectionResultsLambdaName
  lambda_zip_path = var.AnomalyDetectionResultsLambdaZipPath
  handler = var.AnomalyDetectionResultsLambdaHandler
  runtime = var.AnomalyDetectionLambdaRuntime
  roleArn = module.AnomalyDetectionLambdaRole.Role_arn
  memory_size = var.AnomalyDetectionLambdaMemory
  timeout = var.AnomalyDetectionLambdaTimeout
  tags = local.AnomalyDetectionLambdaTags
  layers = [aws_lambda_layer_version.pandas_lambda_layer.arn,aws_lambda_layer_version.RedshiftConn_layer.arn]
  subnet_id = local.subnet_id
  security_group_id = local.security_group_id
}


# ---------------------------------------------------------------------------------------------------------------------
#create sagemaker policy
# ---------------------------------------------------------------------------------------------------------------------

module "AnomalyDetectionSagemakerPolicy" {
  source = "./modules/policy/AnomalyDetectionSagemaker"
  name = "anomalydetection-sagemaker"
  description = "Policy for AnomalyDetection Sagemaker"
  s3bucket = local.s3bucketName
  tags = local.AnomalyDetectionRoleTags
}

# ---------------------------------------------------------------------------------------------------------------------
#create sagemaker role
# ---------------------------------------------------------------------------------------------------------------------

module "AnomalyDetectionSagemakerRole" {
  source = "./modules/role/AnomalyDetectionSagemaker"
  name = "anomalydetection-sagemaker"
  tags = local.AnomalyDetectionRoleTags
  AnomalyDetectionSagemakerPolicyArn = module.AnomalyDetectionSagemakerPolicy.sagemaker_policy_arn
}


# ---------------------------------------------------------------------------------------------------------------------
#adding pipeline code to S3
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_s3_object" "AnomalyDetectionMLCode" {
  bucket = data.aws_s3_bucket.bucket.id
  key    = "${var.codeFilePath}AnomalyDetectionMLcode.py"
  source = "./modules/AnomalyDetectionML/AnomalyDetectionMLcode.py"
  etag   = filemd5("./modules/AnomalyDetectionML/AnomalyDetectionMLcode.py")
}


# ---------------------------------------------------------------------------------------------------------------------
# Config files to be added in S3
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_s3_object" "Metadata" {
  bucket = data.aws_s3_bucket.bucket.id
  key    = "${var.configFilePath}metadata.yaml"
  source = "./config/metadata.yaml"
  etag   = filemd5("./config/metadata.yaml")
}

resource "aws_s3_object" "WalmartConfig" {
  bucket = data.aws_s3_bucket.bucket.id
  key    = "${var.configFilePath}walmart_config.yaml"
  source = "./config/walmart_config.yaml"
  etag   = filemd5("./config/walmart_config.yaml")
}

resource "aws_s3_object" "ScannerConfig" {
  bucket = data.aws_s3_bucket.bucket.id
  key    = "${var.configFilePath}scanner_config.yaml"
  source = "./config/scanner_config.yaml"
  etag   = filemd5("./config/scanner_config.yaml")
}
