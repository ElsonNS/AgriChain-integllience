import json
import boto3
import joblib
import tempfile

s3 = boto3.client('s3')

def load_model():

    tmp = tempfile.NamedTemporaryFile()

    s3.download_file(
        "agrichain-models",
        "price_model.pkl",
        tmp.name
    )

    return joblib.load(tmp.name)


model = load_model()


def handler(event, context):

    params = event['queryStringParameters']
    day = int(params["day"])

    prediction = model.predict([[day]])

    return {
        "statusCode": 200,
        "body": json.dumps({
            "forecast_price": float(prediction[0])
        })
    }