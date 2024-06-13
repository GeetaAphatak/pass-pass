variable "name" {
  description = "policy Name"
}
# variable "lambda_function_name" {
#   description = "Lambda Function Name"
# }
# variable "lambda_function_arn" {
#   description = "Lambda Function ARN"
# }
variable "description" {
  description = "description of policy"
}
variable "s3bucket" {
  description = "s3 bucket name"
}
variable "tags" {
  type = map(string)
}
