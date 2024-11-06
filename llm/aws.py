import boto3
import json
from ..llm_utils import *
from .abstract import LLM
import time

from dotenv import load_dotenv
load_dotenv()

import os

        
class BedRockLLMs(LLM):
    def __init__(self,
                model_name = "meta.llama3-8b-instruct-v1:0", 
                access_key = None,
                secret_key = None,
                secret_token = None,    
                region_name = "us-west-2"
                 ) -> None:
        super().__init__()
        self.client = boto3.client(service_name='bedrock-runtime', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=secret_token)
        self.model_id = model_name
        self.host = 'cloud'
        self.device = None
    
    def __call__(self, message):
        prompt = convert_format(message)
        if 'anthropic' in self.model_id.lower():
            request = {
                "anthropic_version": "bedrock-2023-05-31",
                # Optional inference parameters:
                "max_tokens": 4096,
                "messages": convert_to_multimodal_format(message, has_system=False),
            }
            response = self.client.invoke_model(body=json.dumps(request), modelId=self.model_id, contentType='application/json')
            return json.loads(response['body'].read().decode('utf-8'))['content'][0]['text']
        
        request = {
            "prompt": prompt,
            # Optional inference parameters:
            "max_gen_len": 4096,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        
        response = self.client.invoke_model(body=json.dumps(request), modelId=self.model_id, contentType='application/json')
        return json.loads(response['body'].read().decode('utf-8'))['generation']
    
if __name__ == "__main__":
    llm  = BedRockLLMs()
    print(llm([{'role':'system','content':'You are a friendly assistant'},{'role':'user','content':'How are you today'}]))