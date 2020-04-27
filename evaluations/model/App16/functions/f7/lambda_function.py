def factorial(n):
    result=1
    for i in range(1,n+1):
        result*=i
    return result

def lambda_handler(event, context):
    factorial(10000)
    if 'para3_index' in event:
        event['para3_index']=event['para3_index']+1
    else:
        event['para3_index']=0
    event['para3_result']=event['para3'][event['para3_index']]
    return {
        'statusCode': 200,
        'body': {"name":"f7","input":event}
    }