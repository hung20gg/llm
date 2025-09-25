from .llm.gemini import Gemini, RotateGemini
from .llm.chatgpt import ChatGPT, OpenAIWrapper, RotateOpenAIWrapper
from .llm.vllm import vLLM
from .llm.abstract import LLM
import os
from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def check_multi_model(model_name):

    guarantee_models = [
        'gpt',
        'gemini',
        'grok',
        'claude',
        'llama4'
    ]

    test = model_name.lower().replace('-', '').replace('_', '')

    for gm in guarantee_models:
        if gm in test:
            return True
    
    if 'vl' in test:
        return True
    
    return False


def _get_llm_wrapper(model_name, **kwargs):

    multimodel = check_multi_model(model_name)

    if 'gpt' in model_name:
        logging.info(f"Using ChatGPT with model {model_name}")
        return ChatGPT(model_name=model_name, multimodal=multimodel, **kwargs)
        
    elif 'gemini' in model_name:
        logging.info(f"Using Gemini with model {model_name}")
        return Gemini(model_name=model_name, random_key='exp' in model_name, multimodal=multimodel, **kwargs)

    logging.info(f"Using OpenAI Wrapper model: {model_name}")

    return OpenAIWrapper(model_name=model_name, multimodal=multimodel, **kwargs)


def _get_host_api_prefix(provider, **kwargs):
    """
    Get the host and API prefix based on the provider.
    """
    host = kwargs.get('host', None)
    api_prefix = kwargs.get('api_prefix', None)

    if provider in ['nim', 'nvidia']:
        host = 'https://integrate.api.nvidia.com/v1'
        api_prefix = 'NVIDIA_API_KEY'

    elif provider == 'groq':
        host = 'https://api.groq.com/openai/v1'
        api_prefix = 'GROQ_API_KEY'

    elif provider == 'deepseek':
        host = 'https://api.deepseek.com/v1'
        api_prefix = 'DEEPSEEK_API_KEY'

    elif provider == 'deepinfra':
        host = 'https://api.deepinfra.com/v1/openai'
        api_prefix = 'DEEPINFRA_API_KEY'

    elif provider == 'vllm':
        host = 'http://localhost:8000' if 'host' not in kwargs else kwargs['host']
        api_prefix = 'VLLM_API_KEY' if 'api_prefix' not in kwargs else kwargs['api_prefix']

    return host, api_prefix


def get_llm_wrapper(model_name, **kwargs):

    if model_name.count(':') == 1:
        provider, model = model_name.split(':')
        logging.info(f"Using {provider} with model {model}")

        system_prompt = True
        non_sys_prompt_model = ['llama3.2', 'llama-3.2', 'gemma']
        for key in non_sys_prompt_model:
            if key in model.lower():
                print(f"Disabling system prompt for model {model}")
                system_prompt = False
                break
        
        if provider in ['openai']:
            logging.info(f"Using ChatGPT with model {model}")
            return ChatGPT(model_name=model, **kwargs)
        
        elif provider in ['gemini', 'google']:
            logging.info(f"Using Gemini with model {model}")
            return Gemini(model_name=model, random_key='exp' in model, system=system_prompt, **kwargs)

        host, api_prefix = _get_host_api_prefix(provider, **kwargs)
        api_key = os.getenv(api_prefix, None)

        return OpenAIWrapper(
            model_name=model,
            host=host,
            api_key=api_key,
            system=system_prompt,
            **kwargs
        )
    else:
        logging.info(f"Using default LLM wrapper for model {model_name}")
        return _get_llm_wrapper(model_name=model_name, **kwargs)


def get_rotate_llm_wrapper(model_name, **kwargs):
    provider = None
    if model_name.count(':') == 1:
        provider, model_name = model_name.split(':')
       
    system_prompt = True
    non_sys_prompt_model = ['llama3.2', 'llama-3.2', 'gemma']
    for key in non_sys_prompt_model:
        if key in model_name.lower():
            print(f"Disabling system prompt for model {model_name}")
            system_prompt = False
            break
    
    if 'gemini' in model_name or provider in ['google', 'gemini']:
        logging.info(f"Using Rotate Gemini with model {model_name}")
        return RotateGemini(model_name=model_name, random_key='exp' in model_name, system=system_prompt, **kwargs)

    else:
        logging.info(f"Using Rotate OpenAI Wrapper model: {model_name}")
        host, api_prefix = _get_host_api_prefix(provider, **kwargs)
        return RotateOpenAIWrapper(model_name=model_name, host=host, api_prefix=api_prefix, system=system_prompt, **kwargs)
