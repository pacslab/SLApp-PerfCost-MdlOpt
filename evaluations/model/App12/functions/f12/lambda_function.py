import boto3
import time

def lambda_handler(event, context):
    s3_client = boto3.client('s3')
    # Download a 20MB file from S3
    with open('/tmp/App12_f12_20MB', 'wb') as data:
        s3_client.download_fileobj('serverlessappperfopt-network-intensive-source-bucket', 'App12_f12_20MB', data)
    time.sleep(0.1)
    #Delete the file on S3
    s3_client.delete_object(
        Bucket='serverlessappperfopt-network-intensive-source-bucket',
        Key='App12_f12_20MB'
    )
    time.sleep(0.1)
    # Upload a 20MB file to S3
    with open('/tmp/App12_f12_20MB', 'rb') as data:
        s3_client.upload_fileobj(data, 'serverlessappperfopt-network-intensive-source-bucket', 'App12_f12_20MB')
    return {
        'statusCode': 200,
        'body': {"name":"f12","input":event}
    }
