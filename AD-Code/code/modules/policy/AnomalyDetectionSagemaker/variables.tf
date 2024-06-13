variable "name" {
  description = "policy Name"
}
variable "description" {
  description = "description of policy"
}
variable "s3bucket" {
  description = "s3 bucket name"
}
variable "tags" {
  type = map(string)
}