import boto3
import time

def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    # Download a 10MB file from S3
    with open('/tmp/App16_f16_10MB', 'wb') as data:
        s3_client.download_fileobj('serverlessappperfopt-network-intensive-source-bucket', 'App16_f16_10MB', data)
    return {
        'statusCode': 200,
        'body': {"name":"f16","input":event}
    }
