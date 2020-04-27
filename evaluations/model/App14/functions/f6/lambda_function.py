import json
from decimal import Decimal, getcontext

def pi(digits):
    getcontext().prec = digits
    getcontext().prec += 2
    three = Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n+na, na+8
        d, da = d+da, da+32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s

def lambda_handler(event, context):
    pi(10000)
    if 'para2_index' in event:
        event['para2_index']=event['para2_index']+1
    else:
        event['para2_index']=0
    event['para2_result']=event['para2'][event['para2_index']]
    return {
        'statusCode': 200,
        'body': {"name":"f6","input":event}
    }
