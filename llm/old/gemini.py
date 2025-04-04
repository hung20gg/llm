import json
import logging
import time

from ...llm_utils import *
from ..abstract import LLM
import random

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from dotenv import load_dotenv
load_dotenv()
import os



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)  
      
class Gemini(LLM):
    def __init__(self, model_name = 'gemini-1.5-flash-002', api_key = None, random_key = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_type = 'gemini'
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
            if random_key:
                all_possible_keys = get_all_api_key('GEMINI_API_KEY')
                api_key = random.choice(all_possible_keys)
        

        genai.configure(api_key=api_key)
        
        self.system_instruction = None
        self.__initiate()
        self.chat_session = None
        
    def __initiate(self, system_instruction=None):
        self.system_instruction = system_instruction
        self.model= genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
            ) 
            
    def __change_system_instruction(self, system_instruction):
        if system_instruction != self.system_instruction:
            self.__initiate(system_instruction)
            
    def stream(self, message, temperature=0.6, keep_history=False, **kwargs):
        
        system_instruction = None
        if len(message) > 0 and message[0]['role'] == 'system':
            system_instruction = message[0]['content']
            self.__initiate(system_instruction)
            message = message[1:]
            
        if isinstance(message, list):
            message = convert_to_gemini_format(message) 

        
        response = self.model.generate_content(message, 
                                               safety_settings=self.safety_settings,
                                               generation_config=genai.types.GenerationConfig(
                                                                    candidate_count=1,
                                                                    max_output_tokens=8192,
                                                                    temperature=temperature,
                                                                    ),
                                               stream=True
                                            )   
        
        for chunk in response:
            yield chunk.text
            
        
        
    def __call__(self, message, temperature=0.4, count_tokens=False):
        
        start = time.time()
        
        system_instruction = None
        if message[0]['role'] == 'system':
            system_instruction = message[0]['content']
            self.__initiate(system_instruction)
            message = message[1:]
            
        if isinstance(message, list):
            message = convert_to_gemini_format(message)
        
        response = self.model.generate_content(message, safety_settings=self.safety_settings,
            generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
                                candidate_count=1,
                            
                                max_output_tokens=8192,
                                temperature=temperature)
            )   
        print(response.usage_metadata)
        
        self.input_token += response.usage_metadata.prompt_token_count
        self.output_token += response.usage_metadata.candidates_token_count
        
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
 