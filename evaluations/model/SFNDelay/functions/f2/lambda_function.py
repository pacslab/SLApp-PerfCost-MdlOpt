import json
import time

def lambda_handler(event, context):
    time.sleep(0.2)
    return {
        'statusCode': 200,
        'body': json.dumps({'msg':'f2 sleep 200ms completed'})
    }
