# ---------------------------------------------------------------------------------------------------------------------
# Create policy for access to bucket
# ---------------------------------------------------------------------------------------------------------------------

# Configure Provider
provider "aws" {
  region = "us-east-1"
}

data "aws_iam_policy_document" "lambda_policy_document" {
  statement {
    actions   = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:HeadObject"]
    resources = ["arn:aws:s3:::${var.s3bucket}","arn:aws:s3:::${var.s3bucket}/*"]
    effect = "Allow"
  }
  statement {
    actions   = ["sagemaker:CreateAction","sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob","sagemaker:ListProcessingJobs",
      "sagemaker:StopProcessingJob", "sagemaker:AddTags", "sagemaker:DeleteTags"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["redshift-data:ExecuteStatement","redshift-data:ListDatabases","redshift:GetClusterCredentials"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
     actions   = ["ssm:*"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["iam:PassRole", "iam:ListRoles", "iam:GetRole"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["states:*"]
    resources = ["*"]
    effect = "Allow"
  }
}


resource "aws_iam_policy" "lambda_policy" {
  name = var.name
  description = var.description
  policy = data.aws_iam_policy_document.lambda_policy_document.json
  #tags = var.tags
}

