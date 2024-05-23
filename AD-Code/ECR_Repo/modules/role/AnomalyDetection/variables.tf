variable "name" {
  description = "Role Name"
}

variable "AnomalyDetectionPolicyArn" {
  description = "ARN of policy"
}
variable "tags" {
  type = map(string)
}