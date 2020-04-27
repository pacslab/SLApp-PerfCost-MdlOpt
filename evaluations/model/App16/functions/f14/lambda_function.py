import json
import hashlib

def lambda_handler(event, context):
    hashlib.pbkdf2_hmac('sha512', b'ServerlessAppPerfOpt', b'salt', 80000)
    
    if 'para5_index' in event:
        event['para5_index']=event['para5_index']+1
    else:
        event['para5_index']=0
    event['para5_result']=event['para5'][event['para5_index']]
    return {
        'statusCode': 200,
        'body': {"name":"f14","input":event}
    }
