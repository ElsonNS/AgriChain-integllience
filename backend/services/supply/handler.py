import json
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('FarmerSupply')


def handler(event, context):

    body = json.loads(event['body'])

    crop = body["crop"]
    farmer_id = body["farmer_id"]
    quantity = body["quantity"]
    location = body["location"]

    table.put_item(
        Item={
            "crop": crop,
            "farmer_id": farmer_id,
            "quantity": quantity,
            "location": location,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Supply recorded"})
    }