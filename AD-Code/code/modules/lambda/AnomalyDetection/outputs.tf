#ARN of Newly created resource
output "lambda_function_invoke_uri" {
  value = aws_lambda_function.lambda.invoke_arn
}

output "lambda_function_arn" {
  value = aws_lambda_function.lambda.arn
}