import json
from ..llm_utils import *
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from dotenv import load_dotenv
load_dotenv()

import os

genai.configure(api_key=os.getenv('GENAI_API_KEY'))

        
class Gemini:
    def __init__(self, model_name = 'gemini-1.5-flash'):
        self.model= genai.GenerativeModel(model_name)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
    def __call__(self, message):
        if isinstance(message, list):
            message = convert_to_gemini_format(message)
        response = self.model.generate_content(message, safety_settings=self.safety_settings,
            generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
                                candidate_count=1,
                            
                                max_output_tokens=40000,
                                temperature=0.3)
            )   
        return response.candidates[0].content.parts[0].text
 