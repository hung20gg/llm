import json
import logging
import time

from ..llm_utils import *
from .abstract import LLM

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
    def __init__(self, model_name = 'gemini-1.5-flash-002', api_key = None, random_key = False, **kwargs):
        super().__init__()
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

        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        
        self.system_instruction = None
        self.chat_session = None
            
    def stream(self, messages, temperature=0.6, **config):
        
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
            
        
        
    def __call__(self, messages, temperature=0.4, tools = [], count_tokens=False, **config):
        
        start = time.time()
        
        system_instruction = None
        if messages[0]['role'] == 'system':
            system_instruction = messages[0]['content']
            messages = messages[1:]
            
        print(self.model_name)

        contents = []
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = convert_to_gemini_format(messages) 
            for msg in messages:
                contents.append(types.Content(role=msg['role'], parts=[types.Part.from_text(text=msg['parts'][0])]))
        else:
            contents = [messages]

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
                                                            
                                                        
            print(response.usage_metadata)
            
            end = time.time()
            logging.info(f"Completion time of {self.model_name}: {end - start}s")
            
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
            logging.error(f"Error: {e}")
            return ''