#ARN of Newly created resource
output "anomalydetection_ecr_role_name" {
  value = aws_iam_role.anomalydetection_role.name
}
output "anomalydetection_ecr_role_arn" {
  value = aws_iam_role.anomalydetection_role.arn
}
