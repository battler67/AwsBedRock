import boto3
import base64
import json
from PIL import Image
import io
bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
response = bedrock.invoke_model(
    modelId="stability.sd3-5-large-v1:0",
    body=json.dumps({
        'prompt': 'A car made out of vegetables.'
    })
)
output_body = json.loads(response['body'].read().decode("utf-8"))
base64_output_image = output_body['images'][0]
image_data = base64.b64decode(base64_output_image)
image = Image.open(io.BytesIO(image_data))
image.save("image.png")