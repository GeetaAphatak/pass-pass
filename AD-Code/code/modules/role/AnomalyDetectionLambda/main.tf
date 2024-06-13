# ------------------------anomalydetectionlambda---------------------------------------------------------------------------------------------
# This module creates the anomalydetectionlambda role
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_iam_role" "lambda_role" {
  name = var.name
  tags = var.tags
  assume_role_policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": "sts:AssumeRole",
        "Principal": {
          "Service": "lambda.amazonaws.com"
        },
        "Effect": "Allow",
        "Sid": ""
      }
    ]
}
EOF
}

# Attaching the custom AnomalyDetectionLambda policy
resource "aws_iam_role_policy_attachment" "attach_AnomalyDetectionLambdaPolicy" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = var.AnomalyDetectionLambdaPolicyArn
}

# Attaching the AWSLambdaVPCAccessExecutionRole managed policy
resource "aws_iam_role_policy_attachment" "AWSLambdaVPCAccessExecutionRole" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}