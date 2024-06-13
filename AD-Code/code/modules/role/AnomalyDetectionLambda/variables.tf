variable "name" {
  description = "Role Name"
}

variable "AnomalyDetectionLambdaPolicyArn" {
  description = "ARN of policy"
}
variable "tags" {
  type = map(string)
}