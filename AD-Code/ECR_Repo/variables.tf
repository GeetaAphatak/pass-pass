variable "environment" {
  type = string
}
locals {
  raw_settings = yamldecode(file("./config/metadata.yaml"))
  kmsarn = local.raw_settings.variables.kmsarn
  s3bucketName = local.raw_settings.variables.s3bucketName
  AnomalyDetectionTags = {
                                access-org = "Keynol"
                                BillTagId = "AnomalyDetection"
                                TagID = "AnomalyDetection-ECR"
                                access-team = "ML"
                                access-department = "ML"
                            }
}