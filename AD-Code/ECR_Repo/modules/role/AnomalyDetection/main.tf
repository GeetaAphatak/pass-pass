# ---------------------------------------------------------------------------------------------------------------------
# This module creates the sxm-edo-edm-anomalydetectionsagemakerecr-dev role
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_iam_role" "anomalydetection_role" {
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

resource "aws_iam_role_policy_attachment" "attach_AnomalyDetectionPolicy" {
  role       = aws_iam_role.anomalydetection_role.name
  policy_arn = var.AnomalyDetectionPolicyArn
}

