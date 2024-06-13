AnomalyDetectionLambdaSourceFile = "modules/lambda/AnomalyDetection/code/AnomalyDetection.py"
AnomalyDetectionLambdaZipPath = "modules/lambda/AnomalyDetection/code/zip/AnomalyDetection.zip"
AnomalyDetectionLambdaHandler ="AnomalyDetection.lambda_handler"
AnomalyDetectionResultsLambdaSourceFile = "modules/lambda/AnomalyDetectionResults/code/AnomalyDetectionResults.py"
AnomalyDetectionResultsLambdaZipPath = "modules/lambda/AnomalyDetectionResults/code/zip/AnomalyDetectionResults.zip"
AnomalyDetectionResultsLambdaHandler ="AnomalyDetectionResults.lambda_handler"
AnomalyDetectionLambdaRuntime = "python3.10"
AnomalyDetectionLambdaMemory = 1024
AnomalyDetectionLambdaTimeout = 150


codeFilePath = "MLcode/sagemaker/"
configFilePath = "MLcode/sagemaker/configs/"
MaxThresholdNoOfSagemakerJobs = 12

