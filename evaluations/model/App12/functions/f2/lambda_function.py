def factorial(n):
    result=1
    for i in range(1,n+1):
        result*=i
    return result

def lambda_handler(event, context):
    factorial(28000)
    
    return {
        'statusCode': 200,
        'body': {"name":"f2","input":event}
    }