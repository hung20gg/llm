from .llm.gemini import Gemini, RotateGemini
from .llm.chatgpt import ChatGPT, OpenAIWrapper
from .llm.vllm import vLLM
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

    return OpenAIWrapper(model_name=model_name, **kwargs)
 