import json
import logging
import time

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))  # Add the parent directory to the path

from llm.llm_utils import (
    get_all_api_key, 
    pil_to_base64, 
    convert_to_gemini_format,
    convert_non_system_prompts
    )
from llm.llm.abstract import LLM

from google import genai
import random



from google.genai.types import HarmCategory, HarmBlockThreshold

from dotenv import load_dotenv
load_dotenv()

import os
from google.genai import types


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)  
      
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
        
        system_instruction = None
        if len(messages) > 0 and messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            messages = messages[1:]
            
        contents = []
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = convert_to_gemini_format(messages) 
            for msg in messages:
                contents.append(types.Content(role=msg['role'], parts=[types.Part.from_text(text=msg['parts'][0])]))
        elif isinstance(messages, str):
            contents = [messages] 

        
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
    def convert_to_gemini_format(contents, has_system=True): 
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
                    parts.append(types.Part.from_bytes(data=pil_to_base64(part['image']), mime_type='image/jpeg'))
                elif part.get('type') == 'bytes':
                    mime_type = part.get('mine_type', 'image/jpeg')
                    parts.append(types.Part.from_bytes(data=part['bytes'], mime_type=mime_type))
                else:
                    type_ = part.get('type')
                    raise ValueError(f"Invalid type: {type_}")
        return parts
        
    def __call__(self, messages, temperature=0.4, tools = [], count_tokens=False, **config):
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        start = time.time()
        
        system_instruction = None
        # print(self.model_name)

        contents = []
        if isinstance(messages, list) and isinstance(messages[0], dict):
            
            if messages[0]['role'] == 'system':
                system_instruction = messages[0]['content']
                messages = messages[1:]

            messages = convert_to_gemini_format(messages) 
            for msg in messages:
                parts = self.convert_to_gemini_format(msg['parts'])

                contents.append(types.Content(role=msg['role'], parts=parts))
        elif isinstance(messages, str):
            contents = [messages]

        else:
            contents = messages

        # Check function calling:
        bool_function = False
        functions = []
        if len(tools) > 0:
            for tool in tools:
                if callable(tool):
                    bool_function = True
                    functions.append(tool)

        if bool_function:
            logging.info(f"Number of functions: {len(tools)}. Callables: {len(functions)}")
            
            
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
            logging.info(f"Model name: {self.model_name}, Completion {end - start:.5f}s, Usage {response.usage_metadata}")
            
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
            logging.error(f"Error with API Key ending {self.__api_key[-5:]} : {e}")
            if self.ignore_quota:
                return ''
            else:
                raise e
        
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
            logging.error(f"Error: {e}")
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

            assert len(self.queue) != self.len_queue, "Client leakage"

            client = self.queue.popleft()
            count += 1

            # Only 1 api key
            if self.len_queue == 1:
                break

            if count % len(self.queue) == 0:
                logging.warning("All clients have reached the maximum rpm. Wait for 10 seconds")
                time.sleep(10)
            if count > max_count:
                break

        self.queue.append(client)
        return client.stream(messages = messages, **kwargs)
             
    def __repr__(self):
        return f"RoutingGemini(model_name={self.model_name}, clients={len(self.__api_keys)})"

    def __str__(self):
        return f"RoutingGemini(model_name={self.model_name}, clients={len(self.__api_keys)})"