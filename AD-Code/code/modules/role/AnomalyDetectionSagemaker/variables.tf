variable "name" {
  description = "Role Name"
}

variable "AnomalyDetectionSagemakerPolicyArn" {
  description = "ARN of policy"
}
variable "tags" {
  type = map(string)
}