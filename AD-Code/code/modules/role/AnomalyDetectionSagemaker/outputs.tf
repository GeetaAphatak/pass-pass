#ARN of Newly created resource
output "Sagemaker_role_name" {
  value = aws_iam_role.sagemaker_role.name
}
output "Sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_role.arn
}
