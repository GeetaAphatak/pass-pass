import boto3
import redshift_connector
import pandas as pd
import json

def lambda_handler(event, context):
    print(event)
    # inference_file = event['Records'][0]['s3']['object']['key']
    predicted_file_path= "predictions/scanner/scanner_data - weekly - predictions.csv"
    file_path_splited = predicted_file_path.split('/')
    data_name = file_path_splited[1]
    file_name = file_path_splited[-1]

    if data_name=="scanner":
        if "daily" in file_name:
            table_name = "scanner_brand_sales_daily"
            train_end_date ='2023-11-30'
        else:
            table_name = "scanner_brand_sales_weekly"
            train_end_date = '2023-12-03'
    elif data_name=="walmart":
        if "daily" in file_name:
            table_name = "walmart_sate_daily"
            train_end_date = '2023-11-30'
        else:
            table_name = "walmart_sate_weekly"
            train_end_date = '2023-11-30'

    # Define your S3 bucket name and the file key
    bucket_name = 'keynol-maverick'
    print("##############################")
    print(predicted_file_path, table_name, train_end_date)

    try:
        # Establish connection
        conn = redshift_connector.connect(
            host='maverick-demo-wg.850506728038.us-east-1.redshift-serverless.amazonaws.com',
            database='dev',
            port=5439,
            user='admin',
            password='Newyork~2024'
        )

        cursor = conn.cursor()
        query = f"DELETE FROM dev.maverick_rpt.{table_name} WHERE time_period > '{train_end_date}';"
        print(query)
        cursor.execute(query)


        s3_client = boto3.client('s3')
        # Fetch the file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=predicted_file_path)
        # Read the file content
        df = pd.read_csv(response['Body'])
        print(df.head())

        for row in df.itertuples():
            insert_query = f"""
                INSERT INTO dev.maverick_rpt.{table_name} 
                (time_period, key1, value, lower_value, upper_value, anomaly, train_dataset_flag, feedback_flag)
                VALUES ('{row.time_period}','{row.key1}',{row.value},{row.lower_value},{row.upper_value},{row.anomaly},{str(row.train_dataset_flag).upper()},{str(row.feedback_flag).upper()})
                """
            cursor.execute(insert_query)
        print(insert_query)
        conn.commit()
        conn.close()
        print("##############################")


        return {
            'statusCode': 200,
            'body': "Success"
        }
    except Exception as e:
        # Return error response
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }