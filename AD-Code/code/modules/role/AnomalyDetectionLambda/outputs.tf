#ARN of Newly created resource
output "Role_name" {
  value = aws_iam_role.lambda_role.name
}
output "Role_arn" {
  value = aws_iam_role.lambda_role.arn
}
