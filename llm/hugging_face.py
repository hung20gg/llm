from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

import json
from ..llm_utils import *
from .abstract import LLM
import time


from dotenv import load_dotenv
load_dotenv()

import os


class CoreLLMs(LLM):
    def __init__(self,
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
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

    def __call__(self, messages, **kwargs):
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
        
