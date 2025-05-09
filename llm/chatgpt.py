import json
from ..llm_utils import *
from .abstract import LLM
import time
import logging

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os
import random

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
    def __init__(self, host, model_name, api_key = None, api_prefix = None, random_key = False, multimodal = False, igrone_quota = True, **kwargs):
        super().__init__(model_name=model_name)
        self.host = host
        self.model_name = model_name
        self.api_key = api_key

        if api_key is None and random_key:
            possible_keys = get_all_api_key(api_prefix)
            api_key = random.choice(possible_keys)

        self.__api_key = str(api_key)
        self.client = OpenAI(api_key=api_key, base_url=host)
        self.multimodal = multimodal
        self.ignore_quota = igrone_quota

        
    def stream(self, messages, temperature = 0.6, **kwargs):
        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        try:
            completion = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                stream = True,
                **kwargs
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        except Exception as e:
            if not self.multimodal:
                self.multimodal = True
                self.stream(messages, **kwargs)
            else:
                print(e)
                return ''
                    
    
    def __call__(self, messages, temperature = 0.6, response_format=None, count_tokens=False):
        
        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        start = time.time()
        try:
            completion = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                stream=False
            )
        except Exception as e:

            if not self.multimodal:
                logging.warning("Switching to multimodal")
                self.multimodal = True
                return self(messages, temperature, response_format, count_tokens)

            else:
                
                # Handle edge cases
                logging.error(f"Error with API Key ending {self.__api_key[-5:]} : {e}")
                if self.ignore_quota:
                    return ''
                else:
                    raise e

        if hasattr(completion, 'usage'):
            print(completion.usage)
            
            self.input_token += completion.usage.prompt_tokens
            self.output_token += completion.usage.completion_tokens
        end = time.time()
        logging.info(f"Completion time of {self.model_name}: {end - start}s")

        
        return output_with_usage(completion.choices[0].message.content, completion.usage, count_tokens)
    
class ChatGPT(LLM):
    def __init__(self, model_name = 'gpt-4o-mini', engine='davinci-codex', max_tokens=16384, api_key = None, random_key = False, multimodal = False, ignore_quota = True, **kwargs):
        super().__init__(model_name=model_name)
        self.model_name = model_name
        self.engine = engine

        if api_key is None:
            api_key=os.getenv('OPENAI_API_KEY')
        elif random_key:
            possible_keys = get_all_api_key('OPENAI_API_KEY')
            api_key = random.choice(possible_keys)

        self.__api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model_token = max_tokens
        self.max_tokens = min(self.model_token, max_tokens)
        self.multimodal = multimodal
        self.ignore_quota = ignore_quota
        
    def stream(self, messages, temperature = 0.6, **kwargs):
        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        start = time.time()
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature=temperature,
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
        
            

    def __call__(self, messages, temperature = 0.4, response_format=None, count_tokens=False, tools = None, **kwargs):
        
        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        try:
            start = time.time()
            if response_format is not None: 
                completion = self.client.beta.chat.completions.parse(
                    model = self.model_name,
                    messages = messages,
                    response_format = response_format,
                    stream = False,
                    **kwargs
                )
            else:
                completion = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    temperature=temperature,
                    stream = False,
                    tools = tools,
                    **kwargs
                )    
            
            response = completion.choices[0].message
            
            logging.info(completion.usage)
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
            
            # Function calling
            elif isinstance(tools, list) and len(tools) > 0:
                return output_with_usage(response.tool_calls, completion.usage, count_tokens)
            
            return output_with_usage(response.content, completion.usage, count_tokens)
        
        except Exception as e:
            # Handle edge cases
            logging.error(f"Error with API Key ending {self.__api_key[-5:]} : {e}")
            if self.ignore_quota:
                return ''
            else:
                raise e
    
    def batch_call(self, list_messages, transform = True, prefix = '', example_per_batch=100, sleep_time=10, sleep_step=10):   
        
        """
        Batch call for ChatGPT
        list_messages: list of messages or custom batch
        transform: whether to transform the list of messages into batch
        prefix: prefix for the batch file
        example_per_batch: number of messages per batch
        """

        if transform:
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
                logging.info(f"Sleeping for {sleep_time} seconds")
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
            logging.info(f"Batch {i} created")
            with open(f'batch/batch-{prefix}.jsonl', 'a', encoding='utf-8') as file:
                file.write(json.dumps({"id": batch_job.id}))
                file.write('\n')
    
    def recall_local_batch(self, prefix = '' ):
        batch_ids = []
        if os.path.exists(f'batch/batch-{prefix}.jsonl'):
            with open(f'batch/batch-{prefix}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    batch_id = json.loads(line)['id']

                    batch = self.retrieve(batch_id)


                    logging.info(f"Batch {batch_id} status: {batch.status}")
                    batch_ids.append({
                        "id": batch_id,
                        "status": batch.status
                    })
        return batch_ids

    def recall_online_batch(self, prefix = ''):
        batch_ids = []
        batches = self.client.batches.list()
        for batch in batches:
                logging.info(f"Batch {batch.id} status: {batch.status}")
                batch_ids.append({
                    "id": batch.id,
                    "status": batch.status
                })
        return batch_ids

    def get_successful_messages(self, batch_ids: list[dict]):
        """
        Get successful messages from batch
        Batch_ids: list of batch ids (from recall_local_batch or recall_online_batch)
        """

        successful_messages = []
        for batch_id in batch_ids:
            batch = self.retrieve(batch_id["id"])
            if batch.status == 'completed':
                if batch.output_file_id:
                    file_response = self.client.files.content(batch.output_file_id).text
                    
                    # Save the response to a file
                    with open(f'batch/response-{batch_id["id"]}.jsonl', 'w', encoding='utf-8') as file:
                        file.write(file_response)
                    
                    with open(f'batch/response-{batch_id["id"]}.jsonl', 'r', encoding='utf-8') as file:
                        for line in file:
                            messages = json.loads(line)
                            messages_obj = {
                                "ids": messages.get('custom_id'),
                                "response": messages.get('response').get('body').get('choices')[0].get('message').get('content')
                            }
                            successful_messages.append(messages_obj)

        return successful_messages

    def retrieve(self, batch_id):
        return self.client.batches.retrieve(batch_id)
    

from collections import deque

class ClientOpenAIWrapper:
    def __init__(self, host: str, model_name: str, api_key: str, rpm: int, **kwargs):
        self.llm = OpenAIWrapper(host = host, model_name=model_name, api_key=api_key, **kwargs)
        self.current_request = 0
        self.rpm = rpm
        self.request_time = deque(maxlen=rpm)

    def check_max_rpm(self):
        current_time = time.time() 
        last_1_min = current_time - 60
        if len(self.request_time) == self.rpm:
            begin_request = self.request_time[0]

            # Check if the first request is older than 1 min
            if begin_request <= last_1_min:
                while len(self.request_time) > 0 and self.request_time[0] <= last_1_min:
                    self.request_time.popleft()
                    if len(self.request_time) == 0:
                        self.request_time.append(current_time)
                        return False
                
                self.request_time.append(current_time)
                return False
            else:
                return True
        else:
            self.request_time.append(current_time)
            return False

    def __call__(self, *args, **kwargs):
        return self.llm(*args, **kwargs)
    
    def stream(self,*args, **kwargs):
        return self.llm.stream(*args, **kwargs)

    def __str___(self):
        return f"ClientOpenAIWarrper(model_name={self.llm.model_name}, rpm={self.rpm})"
    

class RotateOpenAIWrapper:


    def __init__(self, host: str, model_name: str, api_keys: list[str] = None, api_prefix = None, rpm: int = 10, **kwargs):
        self._initialized = True
        self.model_name = model_name
        if not api_keys:
            api_keys = get_all_api_key(api_prefix)
        self.__api_keys = api_keys
        assert len(self.__api_keys) > 0, "No api keys found"

        # Randomize the api_keys
        random.shuffle(self.__api_keys)
        self.queue = deque()
        for api_key in self.__api_keys:
            self.queue.append(ClientOpenAIWrapper(host=host, model_name=model_name, api_key=api_key, rpm = rpm, **kwargs))

    def try_request(self, client,  **kwargs):
        try:
            print(client)
            return client( **kwargs), True
        except Exception as e:
            logging.error(f"Error: {e}")
            return '', False

    def __call__(self, messages, **kwargs):
        # Check if the first client in the queue has reached the maximum rpm
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client) # Add back to the queue
            client = self.queue.popleft() # Get the next client
            count += 1
            if count % len(self.queue) == 0:
                logging.warning("All clients have reached the maximum rpm. Wait for 30 seconds")
                time.sleep(30)
            if count > max_count:
                break
        
        # If the client has reached the maximum rpm, add back to the queue and try to make a request
        self.queue.append(client)

        response, success = self.try_request(client, messages = messages, **kwargs)
        
        tries = 0
        while not success:
            client = self.queue.popleft()

            # Check if the client has reached the maximum rpm
            if client.check_max_rpm():
                self.queue.append(client)
                continue

            # If the client has reached the maximum rpm, add back to the queue and try to make a request
            response, success = self.try_request(client, messages = messages, **kwargs)
            tries += 1
            time.sleep(min(tries, 10))
            self.queue.append(client)
            if tries > len(self.queue) * 2:
                logging.error("All clients have failed to make a request")
                return ''
            
        return response
        
    def stream(self, messages, **kwargs):
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client)
            client = self.queue.popleft()
            count += 1
            if count % len(self.queue) == 0:
                logging.warning("All clients have reached the maximum rpm. Wait for 10 seconds")
                time.sleep(10)
            if count > max_count:
                break

        self.queue.append(client)
        return client.stream(messages = messages, **kwargs)
             
    def __repr__(self):
        return f"RoutingOpenAIWapper(model_name={self.model_name}, clients={len(self.__api_keys)})"
    
    def __str__(self):
        return f"RoutingOpenAIWapper(model_name={self.model_name}, clients={len(self.__api_keys)})"