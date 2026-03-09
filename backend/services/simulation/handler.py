import json
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('FarmerSupply')


def handler(event, context):

    crop = event["queryStringParameters"]["crop"]

    response = table.query(
        KeyConditionExpression="crop = :c",
        ExpressionAttributeValues={":c": crop}
    )

    total = sum([item["quantity"] for item in response["Items"]])

    bulk_bonus = total * 0.05

    return {
        "statusCode": 200,
        "body": json.dumps({
            "total_supply": total,
            "estimated_bonus": bulk_bonus
        })
    }