from .llm.gemini import Gemini, RotateGemini
from .llm.chatgpt import ChatGPT, OpenAIWrapper, RotateOpenAIWrapper
from .llm.vllm import vLLM
from .llm.abstract import LLM
import os

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_llm_wrapper(model_name, **kwargs):


    if 'gpt' in model_name:
        logging.info(f"Using ChatGPT with model {model_name}")
        return ChatGPT(model_name=model_name, **kwargs)
        
    elif 'gemini' in model_name:
        logging.info(f"Using Gemini with model {model_name}")
        return Gemini(model_name=model_name, random_key='exp' in model_name, **kwargs)

    logging.info(f"Using OpenAI Wrapper model: {model_name}")

    return OpenAIWrapper(model_name=model_name,  **kwargs)


def get_rotate_llm_wrapper(model_name, **kwargs):
    
    if 'gemini' in model_name:
        logging.info(f"Using Rotate Gemini with model {model_name}")
        return RotateGemini(model_name=model_name, random_key='exp' in model_name, **kwargs)
    
    else:
        logging.info(f"Using Rotate OpenAI Wrapper model: {model_name}")
        return RotateOpenAIWrapper(model_name=model_name,  **kwargs)
 