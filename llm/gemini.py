import json
import time
import os 
import sys
import random
from uuid import uuid4
from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))  # Add the parent directory to the path

from llm.llm_utils import (
    get_all_api_key, 
    pil_to_base64, 
    convert_messages_to_gemini_format,
    convert_non_system_prompts,
    logger
    )
from llm.llm.abstract import LLM
from dotenv import load_dotenv
load_dotenv()

      
class Gemini(LLM):
    def __init__(self, model_name = 'gemini-2.0-flash', api_key = None, random_key = False, igrone_quota = True, system = True, **kwargs):
        super().__init__(model_name=model_name)
        self.model_name = model_name
        self.model_type = 'gemini'
        
        self.safety_settings = [
            types.SafetySetting(
                category = 'HARM_CATEGORY_HATE_SPEECH',
                threshold = 'BLOCK_NONE'
            ),
            types.SafetySetting(
                category = 'HARM_CATEGORY_HARASSMENT',
                threshold = 'BLOCK_NONE'
            ),
            types.SafetySetting(
                category = 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold = 'BLOCK_NONE'
            ),
            types.SafetySetting(
                category = 'HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold = 'BLOCK_NONE'
            ),
        ]

        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        if random_key:
            all_possible_keys = get_all_api_key('GEMINI_API_KEY')
            api_key = random.choice(all_possible_keys)

        self.__api_key = str(api_key)
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        
        self.system_instruction = None
        self.chat_session = None
        self.ignore_quota = igrone_quota
        self.system = system
            
    def stream(self, messages, temperature=0.8, **config):
        if not self.system:
            messages = convert_non_system_prompts(messages)
        
        gemini_conversion = self.convert_conversation_to_gemini_format(messages)
        contents = gemini_conversion['contents']
        system_instruction = gemini_conversion['generation_config']['system_instruction']
        
        response = self.client.models.generate_content_stream(model = self.model_name,
                                                            contents = contents, 
                                                            config=genai.types.GenerateContentConfig(
                                                                    candidate_count=1,
                                                                    max_output_tokens=8192,
                                                                    temperature=temperature,
                                                                    safety_settings=self.safety_settings,
                                                                    system_instruction=system_instruction,
                                                                    **config
                                                                    ),
                                                            )   
        
        for chunk in response:
            yield chunk.text
            
    @staticmethod 
    def _convert_component_to_gemini_format(contents): 
        parts = []
        for part in contents:
            if isinstance(part, str):
                parts.append(types.Part.from_text(text=part))
            elif isinstance(part, dict):
                if part.get('type') == 'text':
                    parts.append(types.Part.from_text(text=part['text']))
                elif part.get('type') == 'image_url':
                    parts.append(types.Part.from_uri(file_uri=part['image_url']['url'], mime_type='image/jpeg'))
                elif part.get('type') == 'image':

                    # Encoded image to base64 url, format "data:image/jpeg;base64,<base64_data>"
                    if isinstance(part['image'], str):
                        if ',' not in part['image'] or not part['image'].startswith('data:image'):
                            raise ValueError("Image string must be a base64 data URL")
                        metadata, encoded = part['image'].split(',', 1)
                        mime_type = metadata.split(';')[0].split(':')[1]
                        parts.append(types.Part.from_bytes(data=encoded, mime_type=mime_type))
                    
                    # Raw image
                    else:
                        parts.append(types.Part.from_bytes(data=pil_to_base64(part['image']), mime_type='image/jpeg'))
                elif part.get('type') == 'bytes':
                    mime_type = part.get('mine_type', 'image/jpeg')
                    parts.append(types.Part.from_bytes(data=part['bytes'], mime_type=mime_type))
                else:
                    type_ = part.get('type')
                    raise ValueError(f"Invalid type: {type_}")
        return parts

    @staticmethod
    def _convert_component_to_gemini_format_json(contents): 
        parts = []
        for part in contents:
            if isinstance(part, str):
                parts.append({'text': part})
            elif isinstance(part, dict):
                if part.get('type') == 'text':
                    parts.append({'text': part['text']})
                elif part.get('type') == 'image_url':
                    parts.append({'image_url': {'url': part['image_url']['url']}})
                elif part.get('type') == 'image':
                    # Encoded image to base64 url, format "data:image/jpeg;base64,<base64_data>"
                    if isinstance(part['image'], str):
                        if ',' not in part['image'] or not part['image'].startswith('data:image'):
                            raise ValueError("Image string must be a base64 data URL")
                        parts.append({'image': part['image']})
                    # Raw image
                    else:
                        parts.append({'image': pil_to_base64(part['image'])})
                elif part.get('type') == 'bytes':
                    mime_type = part.get('mine_type', 'image/jpeg')
                    parts.append({'bytes': part['bytes'], 'mine_type': mime_type})
                else:
                    type_ = part.get('type')
                    raise ValueError(f"Invalid type: {type_}")
        return parts 
    
    def _convert_messages_to_gemini_format_with_object(self, messages):
        
        contents = []
        messages = convert_messages_to_gemini_format(messages) 
        for msg in messages:
            parts = self._convert_component_to_gemini_format(msg['parts'])
            contents.append(types.Content(role=msg['role'], parts=parts))

        return contents
    
    def _convert_messages_to_gemini_format_with_json(self, messages):
        contents = []
        messages = convert_messages_to_gemini_format(messages) 
        for msg in messages:
            parts = self._convert_component_to_gemini_format_json(msg['parts'])
            contents.append({'role': msg['role'], 'parts': parts})
        return contents

    def convert_conversation_to_gemini_format(self, messages, json_format=False):
        system_instruction = None
        if isinstance(messages, list) and isinstance(messages[0], dict):
            
            if messages[0]['role'] == 'system':
                if json_format:
                    system_instruction = {
                        'parts': [{'text': messages[0]['content']}]
                    }
                else:
                    system_instruction = [types.Part.from_text(text=messages[0]['content'])]
                messages = messages[1:]

            if json_format:
                contents = self._convert_messages_to_gemini_format_with_json(messages)
            else:
                contents = self._convert_messages_to_gemini_format_with_object(messages)
            
        elif isinstance(messages, str):
            # For string messages, convert to proper format
            if json_format:
                contents = [{'role': 'user', 'parts': [{'text': messages}]}]
            else:
                contents = [messages]

        else:
            contents = messages

        return {
            'contents': contents,
            'generation_config': {
                'system_instruction': system_instruction
            }
        }
    
    
    def __call__(self, messages, temperature=0.4, tools = [], count_tokens=False, json_format=False, **config):
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        start = time.time()

        gemini_conversion = self.convert_conversation_to_gemini_format(messages, json_format=json_format)
        contents = gemini_conversion['contents']
        system_instruction = gemini_conversion['generation_config']['system_instruction']

        # Check function calling:
        bool_function = False
        functions = []
        if len(tools) > 0:
            for tool in tools:
                if callable(tool):
                    bool_function = True
                    functions.append(tool)

        if bool_function:
            logger.info(f"Number of functions: {len(tools)}. Callables: {len(functions)}")
            
            
            config = types.GenerateContentConfig(
                tools = functions,
                system_instruction=system_instruction,
                safety_settings=self.safety_settings,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode='ANY')
                ),
                **config
            )
        else:
            config=types.GenerateContentConfig(
                candidate_count=1,
                system_instruction=system_instruction,
                safety_settings=self.safety_settings,
                max_output_tokens=8192,
                temperature=temperature,
                **config
            )

        try:
            response = self.client.models.generate_content(model = self.model_name, 
                                                        contents = contents, 
                                                        config=config)
                                                                                                           
            
            end = time.time()
            logger.info(f"Model name: {self.model_name}, Completion {end - start:.5f}s, Usage {response.usage_metadata}")
            
            if count_tokens:
                return {
                    "response": response.candidates[0].content.parts[0].text,
                    "input_token": response.usage_metadata.prompt_token_count,
                    "output_token": response.usage_metadata.candidates_token_count,
                    "total_token": response.usage_metadata.total_token_count
                }
            else:
                return response.candidates[0].content.parts[0].text
        except Exception as e:


            # Log the error
            logger.error(f"Error with API Key ending {self.__api_key[-5:]} : {e}")
            if self.ignore_quota:
                return ''
            else:
                raise e

    
    def batch_call(self, list_messages, key_list=[], temperature=0.4, prefix = '', example_per_batch=100, sleep_time=10, sleep_step=10, **config):
        requests = []

        if not key_list or len(key_list) == 0:
            key_list = []
            for _ in list_messages:
                key_list.append(str(uuid4()))

        assert len(list_messages) == len(key_list), "Length of messages_list and key_list must be the same"
        
        for messages, key in zip(list_messages, key_list):
            messages = convert_non_system_prompts(messages)
            conversation = self.convert_conversation_to_gemini_format(messages, json_format=True)
            conversation['generation_config'] = {
                'temperature': temperature,
                **config
            }
            
            request = {
                "key": key,
                "request": conversation
            }

            requests.append(request)

        batch_requests = []
        for i in range(0, len(requests), example_per_batch):
            batch_requests.append(requests[i:i + example_per_batch])

        # Upload batch requests
        if not os.path.exists('process-gemini'):
            os.mkdir('process-gemini')
        
        if not os.path.exists('batch-gemini'):
            os.mkdir('batch-gemini')

        for i, batch in enumerate(batch_requests):
            process_file = f'process-gemini/process-{prefix}-{i}.jsonl'
            
            # Save file to upload
            for res in batch:
                with open(process_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

            batch_input_file = self.client.files.upload(
                file = process_file,
                config = types.UploadFileConfig(display_name=f'process-{prefix}-{i}', mime_type='jsonl')
            )

            logger.debug(f"Uploaded file: {batch_input_file.name}")

            batch_job = self.client.batches.create(
                model=self.model_name,
                src=batch_input_file.name,
                config={
                    'display_name': f"upload-job-{prefix}-{i}",
                },
            )

            logger.debug(f"Created batch job: {batch_job}")

            with open(f'batch-gemini/batch-{prefix}.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'name': batch_job.name,
                }, ensure_ascii=False) + "\n")

            # Remove the process file after uploading
            if os.path.exists(process_file):
                os.remove(process_file)
                logger.info(f"Removed process file: {process_file}")

            
            
    def retrieve(self, batch_name):
        return self.client.batches.get(name=batch_name)
  
    
    def recall_local_batch(self, prefix = '' ):
        batch_ids = []
        if os.path.exists(f'batch-gemini/batch-{prefix}.jsonl'):
            with open(f'batch-gemini/batch-{prefix}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    batch_name = json.loads(line)['name']

                    batch = self.retrieve(batch_name)


                    logger.info(f"Batch {batch_name} status: {batch.state.name}")
                    batch_ids.append({
                        "name": batch_name,
                        "status": batch.state.name
                    })
        return batch_ids
        
    def get_successful_messages(self, batch_ids = None) -> list[dict]:
        if batch_ids is None:
            batch_ids = self.recall_local_batch()
        
        successful_messages = []
        for batch in batch_ids:
            if batch['status'] == 'JOB_STATE_SUCCEEDED':
                batch_job = self.retrieve(batch['name'])
                result_file_name = batch_job.dest.file_name
                
                file_content_bytes = self.client.files.download(file=result_file_name)
                file_content = file_content_bytes.decode('utf-8')
                # The result file is also a JSONL file. Parse and print each line.
                for line in file_content.splitlines():
                    if line:
                        parsed_response = json.loads(line)
                        
                        key = parsed_response['key']
                        response = parsed_response.get('response', None)
                        if response:
                            text = ''
                            for part in parsed_response['response']['candidates'][0]['content']['parts']:
                                if part.get('text'):
                                    text += part['text']
                            
                            successful_messages.append({
                                "key": key,
                                "text": text
                            })


        return successful_messages

        
        
from collections import deque

class ClientGemini:
    def __init__(self, model_name: str, api_key: str, rpm: int, **kwargs):
        self.llm = Gemini(model_name=model_name, api_key=api_key, **kwargs)
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
        return f"ClientGemini(model_name={self.llm.model_name}, rpm={self.rpm})"

        


class RotateGemini(LLM):
    """ 
    Using queue to call Gemini models in a round-robin fashion
    """

    def __init__(self, model_name = 'gemini-2.0-flash', api_keys: list[str] = None, rpm: int = 15,  **kwargs):
        
        super().__init__(model_name=model_name)

        # Check if the instance has already been initialized
        self._initialized = True
        # Normal init
        self.model_name = model_name
        if not api_keys:
            api_keys = get_all_api_key('GEMINI_API_KEY')
        self.__api_keys = api_keys
        assert len(self.__api_keys) > 0, "No api keys found"

        # Randomize the api_keys
        random.shuffle(self.__api_keys)

        self.queue = deque()
        for api_key in api_keys:
            self.queue.append(ClientGemini(model_name=model_name, api_key=api_key, rpm = rpm, igrone_quota = False, **kwargs))

        self.len_queue = len(self.queue)

        
    def try_request(self, client,  **kwargs):
        try:
            print(client)
            return client(**kwargs), True
        except Exception as e:
            logger.error(f"Error: {e}")
            return '', False

    def __call__(self, messages, **kwargs):
        # Check if the first client in the queue has reached the maximum rpm
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2

        assert len(self.queue) != self.len_queue, "Client leakage"

        while client.check_max_rpm():
            self.queue.append(client) # Add back to the queue
            client = self.queue.popleft() # Get the next client
            count += 1

            if self.len_queue == 1:
                break

            if count % len(self.queue) == 0:
                logger.warning("All clients have reached the maximum rpm. Wait for 30 seconds")
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
                logger.error("All clients have failed to make a request")
                return ''
            
        return response
        
    def stream(self, messages, **kwargs):
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client)

            assert len(self.queue) != self.len_queue, "Client leakage"

            client = self.queue.popleft()
            count += 1

            # Only 1 api key
            if self.len_queue == 1:
                break

            if count % len(self.queue) == 0:
                logger.warning("All clients have reached the maximum rpm. Wait for 10 seconds")
                time.sleep(10)
            if count > max_count:
                break

        self.queue.append(client)
        return client.stream(messages = messages, **kwargs)
             
    def __repr__(self):
        return f"RoutingGemini(model_name={self.model_name}, clients={len(self.__api_keys)})"

    def __str__(self):
        return f"RoutingGemini(model_name={self.model_name}, clients={len(self.__api_keys)})"