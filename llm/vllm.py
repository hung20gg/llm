

from ..llm_utils import *
from .abstract import LLM

import time
import logging

from dotenv import load_dotenv
load_dotenv()

import os
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class vLLM(LLM):
    
    def __init__(self, model_name: str, lora_path: str = None, multimodal = False, **kwargs):
        try:
            # Lazy import of the vllm library
            from vllm import LLM as _vLLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "The 'vllm' library is required to use the vLLM class. "
                "Please install it with 'pip install vllm'."
            ) from e

        super().__init__(model_name=model_name)

        self.model_name = model_name
        self.multimodal = multimodal

        if os.path.exists(lora_path):
            logging.info(f"Loading LoRA model from {lora_path}")
        else:
            logging.warning(f"LoRA model path {lora_path} does not exist. Using default model.")
            lora_path = None

        
        if lora_path:

            from vllm.lora.request import LoRARequest

            self.client = _vLLM(model_name=model_name, enable_lora = True, max_lora_rank=64, **kwargs)
            self.lora_request = LoRARequest("vi_llm", 1, lora_path)
        else:
            self.client = _vLLM(model_name=model_name, **kwargs)
            self.lora_request = None

    def __call__(self, messages, temperature = 0.6, **kwargs):
        if self.multimodal:
            messages = convert_to_multimodal_format(messages)


        start = time.time()
        sampling_params = SamplingParams(temperature=temperature)
        output = self.client.chat(
            messages,
            sampling_params,
            lora_request=self.lora_request,
        )

        generated_text = output.outputs[0].text
        end = time.time()
        logging.info(f"Model name: {self.model_name} Time taken: {end - start:.2f} seconds")
        return generated_text
    

    def stream(self, messages, temperature = 0.6, **kwargs):
        logging.warning("vLLM Offline mode does not support streaming yet.")
        yield self.__call__(messages, temperature, **kwargs)
    

    def batch_call(self, messages, temperature = 0.6, **kwargs):
        if self.multimodal:
            messages = [convert_to_multimodal_format(message) for message in messages]

        start = time.time()
        sampling_params = SamplingParams(temperature=temperature)
        outputs = self.client.chat(
            messages,
            sampling_params,
            lora_request=self.lora_request,
            use_tqdm=True
        )

        generated_texts = [output.outputs[0].text for output in outputs]
        end = time.time()
        logging.info(f"Model name: {self.model_name} Time taken: {end - start:.2f} seconds")
        return generated_texts
        