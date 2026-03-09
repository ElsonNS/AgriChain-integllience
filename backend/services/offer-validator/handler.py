import json

def handler(event, context):

    body = json.loads(event["body"])

    forecast_price = body["forecast_price"]
    buyer_price = body["buyer_price"]

    if buyer_price >= forecast_price * 0.95:
        verdict = "Fair"
    else:
        verdict = "Undervalued"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "verdict": verdict
        })
    }