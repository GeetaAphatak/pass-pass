#ARN of Newly created resource
output "sagemaker_policy_arn" {
  value = aws_iam_policy.sagemaker_policy.arn
}
