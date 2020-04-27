import os

def lambda_handler(event, context):
    path = '/tmp/1MB'
    file_indicator=os.path.isfile(path)
    if file_indicator:
        os.remove(path)
    for i in range(50):
        f = open(path, 'wb')
        f.write(os.urandom(1048576))
        f.flush() 
        os.fsync(f.fileno()) 
        f.close() 

    
    return {
        'statusCode': 200,
        'body': {"name":"f1","input":event}
    }
