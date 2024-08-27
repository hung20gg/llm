from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
import boto3
import json
from llm.llm_utils import *
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

import os

genai.configure(api_key=os.getenv('GENAI_API_KEY'))


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
    

class CoreLLMs:
    def __init__(self,
                model_name = "meta-llama/Meta-Llama-3-8B-Instruct", 
                quantization='auto',
                
                generation_args = None,
                device = None
                ) -> None:
        
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.quantization = quantization
        self.is_agent_initialized = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.host = 'local'
        self._initialize_agent()
        if generation_args is None:

                self.generation_args = {
                    "max_new_tokens": 4096,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "do_sample": True,
                }
        else:
            self.generation_args = generation_args
        
    def _initialize_agent(self):
        
        quantize_config = None
        
        if self.quantization == 'int4':
            print("Using int4")
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
        elif self.quantization == 'int8':
            print("Using int8")
            quantize_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map = self.device, 
            torch_dtype = 'auto', 
            trust_remote_code = True,
            quantization_config = quantize_config 
        )
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        self.pipe = pipeline("text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            trust_remote_code=True,
            device_map=self.device
        )
        
    def _delete_agent(self):
        if self.is_agent_initialized:
            del self.pipe
            del self.model
            gc.collect()
            torch.cuda.empty_cache() 
            self.is_agent_initialized = False
            print("Agent deleted")

    def __call__(self, messages):
        batch_process = False
        if isinstance(messages[0], list):
            print("Batch processing")
            batch_process = True
        if not self.is_agent_initialized:
            self._initialize_agent()
            self.is_agent_initialized = True
        if 'llama' not in self.model_name.lower():
            if batch_process:
                non_sys_prompt_messages = []
                for batch in messages:
                    non_sys_prompt_messages.append(convert_non_system_prompts(batch))
                messages = non_sys_prompt_messages
            else:
                messages = convert_non_system_prompts(messages)
        with torch.no_grad():
            if not batch_process:
                return self.pipe(messages, **self.generation_args)[0]['generated_text'][-1]['content']
            
            results = self.pipe(messages, **self.generation_args)
            return_message = []
            for result in results:
                return_message.append(result[0]['generated_text'][-1]['content'])
            return return_message
        
        
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
        
class BedRockLLMs:
    def __init__(self,
                model_name = "meta.llama3-8b-instruct-v1:0", 
                access_key = None,
                secret_key = None,
                secret_token = None,    
                region_name = "us-west-2"
                 ) -> None:
        self.client = boto3.client(service_name='bedrock-runtime', region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key, aws_session_token=secret_token)
        self.model_id = model_name
        self.host = 'cloud'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def _initialize_agent(self):
        pass
    
    def _delete_agent(self):
        pass
    
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