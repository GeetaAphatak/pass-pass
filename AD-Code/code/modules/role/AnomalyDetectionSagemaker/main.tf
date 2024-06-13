# ---------------------------------------------------------------------------------------------------------------------
# This module creates the anomalydetectionsagemaker role
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_iam_role" "sagemaker_role" {
  name = var.name
  tags = var.tags
  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Principal": {
          "Service": "sagemaker.amazonaws.com"
        },
        "Effect": "Allow",
        "Sid": ""
      }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "attach_AnomalyDetectionSagemakerPolicy" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = var.AnomalyDetectionSagemakerPolicyArn
}

