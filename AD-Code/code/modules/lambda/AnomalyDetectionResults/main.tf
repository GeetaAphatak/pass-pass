# ---------------------------------------------------------------------------------------------------------------------
# This module creates the Anomaly Detection Results lambda function
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_lambda_function" "lambda" {
  function_name = var.name
  filename      = var.lambda_zip_path
  handler = var.handler
  runtime = var.runtime
  memory_size = var.memory_size
  timeout = var.timeout
  role = var.roleArn
  tags = var.tags
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  layers = var.layers
  vpc_config {
    subnet_ids         = var.subnet_id
    security_group_ids = var.security_group_id
  }
  environment {
    variables = {
      subnet_id = join(",",var.subnet_id)
      security_group_id = join(",",var.security_group_id)
    }
  }
}

#resource "aws_lambda_permission" "allow_bucket" {
#  statement_id  = "AllowExecutionFromS3Bucket"
#  action        = "lambda:InvokeFunction"
#  function_name = aws_lambda_function.lambda.arn
#  principal     = "s3.amazonaws.com"
#  source_arn    = "arn:aws:s3:::keynol-maverick"
#}

#
#resource "aws_s3_bucket_notification" "bucket_notification" {
#  bucket = "keynol-maverick"
#  lambda_function {
#    lambda_function_arn = aws_lambda_function.lambda.arn
#    events              = ["s3:ObjectCreated:*"]
#    filter_prefix       = "inference/"
#    filter_suffix       = ".csv"
#  }
#
#  depends_on = [aws_lambda_permission.allow_bucket]
#}
#
# # Attach the policy to the Lambda role
# resource "aws_iam_role_policy_attachment" "lambda_attach_policy" {
#   role       = module.AnomalyDetectionLambdaRole.Role_name
#   policy_arn = module.AnomalyDetectionLambdaPolicy.Policy_arn
# }
#
# # ---------------------------------------------------------------------------------------------------------------------
# # Add permission for S3 to invoke the Lambda function
# # ---------------------------------------------------------------------------------------------------------------------
#
# resource "aws_lambda_permission" "s3_invoke_lambda" {
#   statement_id  = "AllowS3InvokeLambda"
#   action        = "lambda:InvokeFunction"
#   function_name = var.name
#   principal     = "s3.amazonaws.com"
#   source_arn    = "arn:aws:s3:::${var.s3bucket}"
# }
#
#







#
#
#
#
# # ---------------------------------------------------------------------------------------------------------------------
# # Add permission for S3 to invoke the Lambda function
# # ---------------------------------------------------------------------------------------------------------------------
#
# # data "aws_iam_policy_document" "lambda_invoke_policy_document" {
# #   statement {
# #     actions = ["lambda:InvokeFunction"]
# #     resources = ["arn:aws:lambda:us-east-1:*:function:${var.lambda_function_name}"]
# #     principals {
# #       type = "Service"
# #       identifiers = ["s3.amazonaws.com"]
# #     }
# #     effect = "Allow"
# #   }
# # }
#
# resource "aws_lambda_permission" "s3_invoke_lambda" {
#   statement_id  = "AllowS3InvokeLambda"
#   action        = "lambda:InvokeFunction"
#   function_name = var.lambda_function_name
#   principal     = "s3.amazonaws.com"
#   source_arn    = "arn:aws:s3:::${var.s3bucket}"
# }
#
# # ---------------------------------------------------------------------------------------------------------------------
# # Configure S3 bucket notification
# # ---------------------------------------------------------------------------------------------------------------------
#
