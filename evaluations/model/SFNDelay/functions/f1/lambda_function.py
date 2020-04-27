import json
import time

def lambda_handler(event, context):
    time.sleep(0.4)
    return {
        'statusCode': 200,
        'body': json.dumps({'msg':'f1 sleep 400ms completed'})
    }
