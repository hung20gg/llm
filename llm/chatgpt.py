import json
from ..llm_utils import *
from .abstract import LLM
import time
import logging

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def output_with_usage(response, usage, count_tokens=False):
    if count_tokens:
        return {
            "response": response,
            "input_token": usage.prompt_tokens,
            "output_token": usage.completion_tokens,
            "total_token": usage.total_tokens
        }
    return response

class OpenAIWrapper(LLM):
    def __init__(self, host, model_name, api_key, **kwargs):
        super().__init__()
        self.host = host
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=host)
        
    def stream(self, message, **kwargs):
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = message,
            stream = True,
            **kwargs
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                    
    
    def __call__(self, messages, temperature = 0.6, response_format=None, count_tokens=False):
        
        start = time.time()
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature=temperature,
            stream=self.stream
        )
        if hasattr(completion, 'usage'):
            print(completion.usage)
            
            self.input_token += completion.usage.prompt_tokens
            self.output_token += completion.usage.completion_tokens
        end = time.time()
        logging.info(f"Completion time of {self.model_name}: {end - start}s")

        
        return output_with_usage(completion.choices[0].message.content, completion.usage, count_tokens)
    
class ChatGPT(LLM):
    def __init__(self, model_name = 'gpt-4o-mini', engine='davinci-codex', max_tokens=16384, stream = False, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.engine = engine
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), )
        self.model_token = max_tokens
        self.max_tokens = min(self.model_token, max_tokens)
        
    def stream(self, message, **kwargs):
        start = time.time()
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = message,
            stream = True,
            stream_options={"include_usage": True},
            **kwargs
        )
        for chunk in completion:
            content = chunk.choices
            if len(content) > 0:
                yield content[0].delta.content
                
        end = time.time()
        logging.info(f"Completion time of {self.model_name}: {end - start}s")
        
            

    def __call__(self, messages, temperature = 0.4, response_format=None, count_tokens=False, **kwargs):
        try:
            start = time.time()
            if response_format is not None: 
                completion = self.client.beta.chat.completions.parse(
                    model = self.model_name,
                    messages = messages,
                    response_format = response_format,
                    stream = False,
                )
            else:
                completion = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    temperature=temperature,
                    stream = False,
                )    
            
            response = completion.choices[0].message
            
            print(completion.usage)
            self.input_token += completion.usage.prompt_tokens
            self.output_token += completion.usage.completion_tokens
            end = time.time()
            logging.info(f"Completion time of {self.model_name}: {end - start}s")

            
            # Return the parsed response if it exists
            if response_format is not None:
                if response.parsed:
                    return output_with_usage(response.parsed, completion.usage, count_tokens)
                elif response.refusal:
                    return output_with_usage(response.refusal, completion.usage, count_tokens)
            return output_with_usage(response.content, completion.usage, count_tokens)
        
        except Exception as e:
            # Handle edge cases
            print(e)
            return None
    
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
    
