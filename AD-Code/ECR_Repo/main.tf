

#  usage note:  run local first then uncomment below and do a terraform init to copy state to bucket
#
#This module requires terraform Version 0.12.0+ and uses S3 for tfstate
terraform {
  required_version = ">= 0.12.0"
  backend "s3" {
    region  = "us-east-1"
    bucket  = "keynol-maverick"
    key     = "s3://keynol-maverick/MLcode/ecr/terraform.tfstate"
    encrypt = "true"
    acl     = "bucket-owner-full-control"
  }
}

# ---------------------------------------------------------------------------------------------------------------------
#create ECR policy
# ---------------------------------------------------------------------------------------------------------------------

module "AnomalyDetectionPolicy" {
  source = "./modules/policy/AnomalyDetection"
  name = "keynol-anomalydetection-ecr"
  description = "Policy for AnomalyDetection ECR"
  s3bucket = local.s3bucketName
  kmsarn = local.kmsarn
  tags = local.AnomalyDetectionTags
}

# ---------------------------------------------------------------------------------------------------------------------
#create ECR role
# ---------------------------------------------------------------------------------------------------------------------
module "AnomalyDetectionRole" {
  source = "./modules/role/AnomalyDetection"
  name = "keynol-anomalydetection-ecr-role"
  tags = local.AnomalyDetectionTags
  AnomalyDetectionPolicyArn = module.AnomalyDetectionPolicy.anomalydetection_ecr_policy_arn
}


# ---------------------------------------------------------------------------------------------------------------------
#create ECR repo
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_ecr_repository" "AdEcrImage" {
  name                 = "keynol-ml-docker-image"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
  tags = local.AnomalyDetectionTags
}

# ---------------------------------------------------------------------------------------------------------------------
#create Sagemaker notebook instance
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_sagemaker_notebook_instance" "ni" {
  name          = "anomalydetection-notebook"
  role_arn      = module.AnomalyDetectionRole.anomalydetection_ecr_role_arn
  instance_type = "ml.t2.medium"
  tags = local.AnomalyDetectionTags
}