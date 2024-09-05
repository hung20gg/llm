import json
from ..llm_utils import *
import time

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os
print(os.getenv('OPENAI_API_KEY'))

class ChatGPT:
    def __init__(self, model_name = 'gpt-4o-mini', engine='davinci-codex', max_tokens=40000):
        self.model_name = model_name
        self.engine = engine
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model_token = 16384 if model_name == 'gpt-4o-mini' else 40000
        self.max_tokens = min(self.model_token, max_tokens)

    def __call__(self, messages):
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages
        )
        return completion.choices[0].message.content
    
    def batch_call(self, list_messages, prefix = '', example_per_batch=100, sleep_time=10, sleep_step=10):   
        
        list_messages = list_of_messages_to_batch_chatgpt(list_messages, example_per_batch=example_per_batch, model_type=self.model_name, prefix=prefix, max_tokens=self.max_tokens)

        if not os.path.exists('process'):
            os.mkdir('process')
        
        if not os.path.exists('batch'):
            os.mkdir('batch')
        
        for i, batch in enumerate(list_messages):
            with open(f'process/process-{prefix}-{i}.jsonl', 'w', encoding='utf-8') as file:
                for message in batch:
                    json_line = json.dumps(message)
                    file.write(json_line + '\n')

        for i, batch in enumerate(list_messages):
            if i % sleep_step == 0 and i != 0 and sleep_time != 0:
                print(f"Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
            batch_input_file = self.client.files.create(file=open(f'process/process-{prefix}-{i}.jsonl', 'rb'), purpose='batch')
            
            batch_input_file_id = batch_input_file.id
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": f"batch_{i}"
                }
            )
            print(f"Batch {i} created")
            with open(f'batch/batch-{prefix}-{i}.json', 'w', encoding='utf-8') as file:
                file.write(json.dumps(batch_job.id, indent=4))
        
    def retrieve(self, batch_id):
        return self.client.batches.retrieve(batch_id)
    