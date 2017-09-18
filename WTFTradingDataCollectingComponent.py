from __future__ import print_function

import os
import json
import boto3

from time import sleep
from datetime import datetime
from urllib2 import urlopen
from botocore.exceptions import ClientError

MARKETDATAURL = os.environ['site']  # URL of the site to check, stored in the site environment variable, e.g. https://aws.amazon.com
EXPECTED = os.environ['expected']  # String expected to be on the page, stored in the expected environment variable, e.g. Amazon
MARKETNAME = os.environ['marketName']
TIMESTAMP = os.environ['timeStamp']

def validateSiteData(rawMarketData):
    checkResult = rawMarketData['success']
    return EXPECTED == str(checkResult)

def uploadDataToDDB(rawMarketData):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('WTFHistoricalCurrencyData')
    
    for record in rawMarketData['result']:
        print('input market ' + record[MARKETNAME])
        table.put_item(
            Item = {
                'MarketName': record[MARKETNAME],
                'TimeStamp': record[TIMESTAMP],
                'High': str(record['High']),
                'Low': str(record['Low']),
                'Volume': str(record['Volume']),
                'Last': str(record['Last']),
                'BaseVolume': str(record['BaseVolume']),
                'Bid': str(record['Bid']),
                'Ask': str(record['Ask']),
                'OpenBuyOrders': str(record['OpenBuyOrders']),
                'OpenSellOrders': str(record['OpenSellOrders']),
                'PrevDay': str(record['PrevDay'])
            }
        )
        sleep(0.2)
    
    return

def lambda_handler(event, context):
    print('Checking {} at {}...'.format(MARKETDATAURL, event['time']))
    try:
        rawMarketData = json.loads(urlopen(MARKETDATAURL).read())
        if not validateSiteData(rawMarketData):
            raise Exception('Validation failed')
        
        uploadDataToDDB(rawMarketData)

    except Exception:
        print('Check failed!')
        raise
    except ClientError as e:
        print(e)
        raise
    else:
        print('Check passed!')
        return event['time']
    finally:
        print('Check complete at {}'.format(str(datetime.now())))
