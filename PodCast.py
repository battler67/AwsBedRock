import boto3
import json

with open('processed_text.txt', 'r') as f:
    text = f.read()

prompt = f"""
You are a podcast host.

Convert the following text into a natural, engaging podcast transcript 
with two speakers (Host and Guest), conversational tone.

Text:
{text}
"""

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

payload = {
    "prompt": prompt,
    "max_gen_len": 512,
    "temperature": 0.7,
    "top_p": 0.9
}

response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId="meta.llama3-8b-instruct-v1:0",
    contentType="application/json"
)

response_body = json.loads(response["body"].read())

print(response_body["generation"])