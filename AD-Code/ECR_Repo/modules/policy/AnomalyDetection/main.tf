# ---------------------------------------------------------------------------------------------------------------------
# Create policy for access to bucket
# ---------------------------------------------------------------------------------------------------------------------

# Configure Provider
provider "aws" {
  region = "us-east-1"
}

data "aws_iam_policy_document" "ecr_policy_document" {
  statement {
    actions   = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.s3bucket}","arn:aws:s3:::${var.s3bucket}/*"]
    effect = "Allow"
  }
  statement {
    actions   = ["sagemaker:CreateAction","sagemaker:CreateProcessingJob","sagemaker:DescribeProcessingJob","sagemaker:ListProcessingJobs","sagemaker:StopProcessingJob","sagemaker:CreateTrainingJob"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["kms:Decrypt","kms:DescribeKey","kms:GenerateDataKey"]
    resources = [var.kmsarn]
    effect = "Allow"
  }
  statement {
    actions   = ["lambda:GetFunction","lambda:ListFunctions","lambda:InvokeFunction"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["iam:GetRole"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["ec2:CreateNetworkInterface",
                "ec2:DescribeDhcpOptions",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DeleteNetworkInterface",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeVpcs",
                "ec2:CreateNetworkInterfacePermission"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["sns:Publish"]
    resources = ["*"]
    effect = "Allow"
  }
  statement {
    actions   = ["ecr:BatchGetImage","ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:CreateRepository",
                "ecr:Describe*",
                "ecr:GetDownloadUrlForLayer",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage"]
    resources = ["*"]
    effect = "Allow"
  }
}

resource "aws_iam_policy" "ecr_policy" {
  name = var.name
  description = var.description
  policy = data.aws_iam_policy_document.ecr_policy_document.json
}

