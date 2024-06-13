#ARN of Newly created resource
output "Policy_arn" {
  value = aws_iam_policy.lambda_policy.arn
}
