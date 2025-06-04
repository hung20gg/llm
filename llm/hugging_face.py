from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel, LoraConfig
import torch
import gc
from threading import Thread

import json
from ..llm_utils import convert_non_system_prompts
from .abstract import LLM
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from dotenv import load_dotenv
load_dotenv()

import os


class HuggingFaceLLM(LLM):
    def __init__(self,
                model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                lora_name = None,
                quantization='auto',
                generation_args = None,
                device = None,
                **kwargs
                ) -> None:
        super().__init__(model_name=model_name)
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.quantization = quantization
        self.is_agent_initialized = True
        self.lora_name = lora_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.host = 'local'
        self._initialize_agent()
        if generation_args is None:

                self.generation_args = {
                    "max_new_tokens": 4096,
                    "temperature": 0.6,
                    "top_p": 0.8,
                }
        else:
            self.generation_args = generation_args
        
    def _initialize_agent(self):
        
        quantize_config = None
        
        if self.quantization == 'int4':
            logging.info("Using int4 quantization with bfloat16 compute dtype")
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
        elif self.quantization == 'int8':
            logging.info("Using int8 quantization with bfloat16 compute dtype")
            quantize_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
                )
        else:
            logging.info("Using full bfloat16 precision (no quantization)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map = self.device, 
            torch_dtype = torch.bfloat16, 
            trust_remote_code = True,
            quantization_config = quantize_config 
        )

        if self.lora_name:
            logging.info(f"Loading LoRA adapter from {self.lora_name}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )


        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        self.pipe = pipeline("text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            trust_remote_code=True,
            device_map=self.device,

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
        
        start = time.time()
        
        batch_process = False
        if isinstance(messages[0], list):
            print("Batch processing")
            batch_process = True
        if not self.is_agent_initialized:
            self._initialize_agent()
            self.is_agent_initialized = True

        # Only Gemma not support system prompt:
        if 'gemma' in self.model_name.lower():
            if batch_process:
                non_sys_prompt_messages = []
                for batch in messages:
                    non_sys_prompt_messages.append(convert_non_system_prompts(batch))
                messages = non_sys_prompt_messages
            else:
                messages = convert_non_system_prompts(messages)

        print(f"Messages: {messages}")
        with torch.no_grad():
            if not batch_process:
                answer = self.pipe(messages, **self.generation_args)[0]['generated_text'][-1]['content']
            
            else:
                results = self.pipe(messages, **self.generation_args)
                answer = []
                for result in results:
                    answer.append(result[0]['generated_text'][-1]['content'])
                
        end = time.time()
        logging.info(f"Completion time of {self.model_name}: {end - start}s")
        return answer
        
    def stream(self, messages, **kwargs):
        """
        Stream tokens from the model one by one.
        
        Args:
            messages: Input messages to process
            **kwargs: Additional arguments for generation
            
        Yields:
            Generated tokens one by one
        """
        start = time.time()
        
        # Check if batch processing is requested
        batch_process = False
        if isinstance(messages[0], list):
            logging.warning("Batch processing not supported for streaming. Using first message only.")
            messages = messages[0]
        
        # Initialize agent if needed
        if not self.is_agent_initialized:
            self._initialize_agent()
            self.is_agent_initialized = True
            
        # Convert messages if not Llama model
        if 'gemma' in self.model_name.lower():
            messages = convert_non_system_prompts(messages)
        
        # Create a TextIteratorStreamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # Prepare generation arguments
        generation_kwargs = self.generation_args.copy()
        if kwargs:
            generation_kwargs.update(kwargs)
            
        # Add streamer to generation args
        generation_kwargs["streamer"] = streamer
        
        # Process input based on format (chat vs text)
        if isinstance(messages, list) and any(isinstance(m, dict) for m in messages):
            # Chat format - use tokenizer's chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            
            # Create explicit attention mask (1 for tokens to attend to, 0 for padding)
            attention_mask = torch.ones_like(prompt)
            
            inputs = {
                "input_ids": prompt.to(self.device),
                "attention_mask": attention_mask.to(self.device)
            }
            
            # Start generation in a separate thread
            thread = Thread(
                target=self.model.generate,
                kwargs={**inputs, **generation_kwargs}
            )
            thread.start()
            
            # Stream tokens as they're generated
            for token in streamer:
                yield token
        else:
            # Text format
            encoded = self.tokenizer(messages, return_tensors="pt", padding=True)
            
            inputs = {
                "input_ids": encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device)
            }
            
            # Start generation in a separate thread
            thread = Thread(
                target=self.model.generate,
                kwargs={**inputs, **generation_kwargs}
            )
            thread.start()
            
            # Stream tokens as they're generated
            for token in streamer:
                yield token
        
        end = time.time()
        logging.info(f"Streaming completion time of {self.model_name}: {end - start}s")
