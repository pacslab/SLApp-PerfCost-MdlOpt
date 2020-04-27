import os

def lambda_handler(event, context):
    path = '/tmp/2MB'
    file_indicator=os.path.isfile(path)
    if file_indicator:
        os.remove(path)
    for i in range(10):
        f = open(path, 'wb')
        f.write(os.urandom(2097152))
        f.flush() 
        os.fsync(f.fileno()) 
        f.close() 

    if 'para6_index' in event:
        event['para6_index']=event['para6_index']+1
    else:
        event['para6_index']=0
    event['para6_result']=event['para6'][event['para6_index']]
    return {
        'statusCode': 200,
        'body': {"name":"f15","input":event}
    }
