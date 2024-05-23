#ARN of Newly created resource
output "anomalydetection_ecr_policy_arn" {
  value = aws_iam_policy.ecr_policy.arn
}
