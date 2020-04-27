import json
import hashlib

def lambda_handler(event, context):
    hashlib.pbkdf2_hmac('sha512', b'ServerlessAppPerfOpt', b'salt', 100000)
    
    return {
        'statusCode': 200,
        'body': json.dumps("f5")
    }
