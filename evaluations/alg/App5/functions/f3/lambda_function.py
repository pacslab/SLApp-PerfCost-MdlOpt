def fibonacci(n):
    if n<=1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    
def lambda_handler(event, context):
    fibonacci(28)
    
    return {
        'statusCode': 200,
        'body': "'f3'"
    }