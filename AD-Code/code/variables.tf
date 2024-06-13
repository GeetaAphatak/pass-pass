variable "AnomalyDetectionLambdaSourceFile" {
  type = string
}
variable "AnomalyDetectionLambdaZipPath" {
  type = string
}
variable "AnomalyDetectionLambdaHandler" {
  type = string
}
variable "AnomalyDetectionResultsLambdaSourceFile" {
  type = string
}
variable "AnomalyDetectionResultsLambdaZipPath" {
  type = string
}
variable "AnomalyDetectionResultsLambdaHandler" {
  type = string
}
variable "AnomalyDetectionLambdaRuntime" {
  type = string
}
variable "AnomalyDetectionLambdaMemory" {
  type = number
}
variable "AnomalyDetectionLambdaTimeout" {
  type = number
}


locals {
  raw_settings = yamldecode(file("./config/metadata.yaml"))
  account-id = local.raw_settings.variables.account-id
  subnet_id = local.raw_settings.variables.subnet_id
  security_group_id = local.raw_settings.variables.security_group_id
  imageUri = local.raw_settings.variables.imageUri
  s3bucketName = local.raw_settings.variables.s3bucketName
  AnomalyDetectionLambdaName = local.raw_settings.variables.AnomalyDetectionLambdaName
  AnomalyDetectionResultsLambdaName = local.raw_settings.variables.AnomalyDetectionResultsLambdaName
  region = local.raw_settings.variables.region
  vpc_endpoint_ids = local.raw_settings.variables.vpc_endpoint_ids
  AnomalyDetectionLambdaTags = {
                                access-org = "Keynol"
                                BillTagId = "AnomalyDetection"
                                TagID = "AnomalyDetection-lambda"
                                access-team = "ML"
                                access-department = "ML"
                            }
  AnomalyDetectionRoleTags = {
                              access-org = "Keynol"
                                BillTagId = "AnomalyDetection"
                                TagID = "AnomalyDetection-role"
                                access-team = "ML"
                                access-department = "ML"
                            }
}


variable "codeFilePath"{
  type = string
}
# variable "modelCodeFilePath"{
#   type = string
# }
variable "configFilePath"{
  type = string
}
variable "MaxThresholdNoOfSagemakerJobs" {
  type = number
}
# variable "stage_name" {
#   type = string
# }
# variable "usage_plan_name"{
#   type = string
# }




