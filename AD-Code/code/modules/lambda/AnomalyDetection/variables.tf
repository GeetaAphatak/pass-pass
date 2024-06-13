variable "name" {
  type = string
}
variable "lambda_zip_path" {
  type = string
}
variable "handler" {
  type = string
}
variable "runtime" {
  type = string
}
variable "memory_size" {
  type = number
}
variable "timeout" {
  type = number
}
variable "roleArn" {

}
variable "tags" {
  type = map(string)
}
variable "layers" {
  type = list(string)
}
variable "subnet_id" {
  type = list(string)
}
variable "security_group_id" {
  type = list(string)
}
# variable "s3bucket" {
#   type = list(string)
# }