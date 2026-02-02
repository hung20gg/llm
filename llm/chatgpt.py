import json
import os 
import sys
import random
import time
from uuid import uuid4
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Dict, Any, Union, Iterator, Generator, Tuple, AsyncIterator
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))  # Add the parent directory to the path

from llm.llm_utils import (
    get_all_api_key, 
    convert_to_multimodal_format, 
    list_of_messages_to_batch_chatgpt,
    convert_non_system_prompts,
    logger
    )
from llm.llm.abstract import LLM


from dotenv import load_dotenv
load_dotenv()

def output_with_usage(response: Any, usage: Any, count_tokens: bool = False) -> Union[Dict[str, Any], Any]:
    if count_tokens:
        return {
            "response": response,
            "input_token": usage.prompt_tokens,
            "output_token": usage.completion_tokens,
            "total_token": usage.total_tokens
        }
    return response


def _openai_text_completion_stream(client: OpenAI, **kwargs: Any) -> Iterator[Optional[Any]]:
    try:
        completion = client.chat.completions.create(
            stream=True,
            **kwargs
        )
        for chunk in completion:
            yield chunk
    except Exception as e:
        logger.error(f"Error in chat completion stream: {e}")
        yield None
        

async def _openai_text_completion_stream_async(client: AsyncOpenAI, **kwargs: Any) -> AsyncIterator[Optional[Any]]:
    try:
        completion = await client.chat.completions.create(
            stream=True,
            **kwargs
        )
        async for chunk in completion:
            yield chunk
    except Exception as e:
        logger.error(f"Error in chat completion stream: {e}")
        

def _openai_text_completion(client: OpenAI, **kwargs: Any) -> Any:
        
    completion = client.chat.completions.create(
        **kwargs
    )
    
    response = completion.choices[0].message    
    logger.info(completion.usage)
    
    # Return the parsed response if it exists
    if kwargs.get('response_format') is not None:
        if response.parsed:
            return response.parsed
        elif response.refusal:
            return response.refusal

    return response.content


async def _openai_text_completion_async(client: AsyncOpenAI, **kwargs: Any) -> Any:
        
    completion = await client.chat.completions.create(
        **kwargs
    )
    
    response = completion.choices[0].message    
    logger.info(completion.usage)
    
    # Return the parsed response if it exists
    if kwargs.get('response_format') is not None:
        if response.parsed:
            return response.parsed
        elif response.refusal:
            return response.refusal

    return response.content


def _openai_tool_calling(client: OpenAI, **kwargs: Any) -> Dict[str, Any]:

    completion  = client.chat.completions.create(
        **kwargs
    )
    content = ""
    tool_calls = []
    if completion.choices[0].message.tool_calls is not None:
        for choice in completion.choices[0].message.tool_calls:
            tool_calls.append(choice.model_dump())
    content = completion.choices[0].message.content

    return {
        "content": content,
        "tool_calls": tool_calls
    }
    
async def _openai_tool_calling_async(client: AsyncOpenAI, **kwargs: Any) -> Dict[str, Any]:
    completion  = await client.chat.completions.create(
        **kwargs
    )
    content = ""
    tool_calls = []
    if completion.choices[0].message.tool_calls is not None:
        for choice in completion.choices[0].message.tool_calls:
            tool_calls.append(choice.model_dump())
    content = completion.choices[0].message.content

    return {
        "content": content,
        "tool_calls": tool_calls
    }


class OpenAIWrapper(LLM):
    def __init__(self, host: str, model_name: str, api_key: Optional[str] = None, api_prefix: Optional[str] = None, random_key: bool = False, multimodal: bool = False, ignore_quota: bool = True, system: bool = True, **kwargs: Any) -> None:
        self.__initiate_client(host, model_name, api_key, api_prefix, random_key, multimodal, ignore_quota, system, **kwargs)
        
        
    def __initiate_client(self, host: str, model_name: str, api_key: Optional[str] = None, api_prefix: Optional[str] = None, random_key: bool = False, multimodal: bool = False, ignore_quota: bool = True, system: bool = True, **kwargs: Any) -> None:
        self.host = host
        self.model_name = model_name
        self.api_key = api_key

        print(f"Initializing OpenAIWrapper with model {model_name} at host {host} and API key {api_key}")

        if api_key is None and random_key:
            print("Selecting random API key")
            if api_prefix is None:
                api_prefix = 'OPENAI_API_KEY'
                
            possible_keys = get_all_api_key(api_prefix)
            api_key = random.choice(possible_keys)

        self._api_key = str(api_key)
        self.client = OpenAI(api_key=api_key, base_url=host)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=host)
        self.multimodal = multimodal
        self.ignore_quota = ignore_quota
        self.system = system

    def stream(self, messages: List[Dict[str, Union[str, List[Dict]]]], temperature: float = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Iterator[Optional[str]]:
        

        # Disable system prompt (for gemma or llama 3.2)
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)    

        try:
            if tools is not None and len(tools) > 0:
                return self.stream_tool_calling(
                    messages = messages,
                    temperature=temperature,
                    tools = tools,
                    **kwargs
                )
            else:
                completion = _openai_text_completion_stream(
                    client = self.client,
                    model = self.model_name,
                    messages = messages,
                    temperature=temperature,
                    **kwargs
                )
            if completion:
                for chunk in completion:
                    if chunk is not None:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return None
        
    
    async def astream(self, messages: List[Dict[str, Union[str, List[Dict]]]], temperature: float = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> AsyncIterator[Optional[str]]:

        # Disable system prompt (for gemma or llama 3.2)
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        try:
            async for chunk in _openai_text_completion_stream_async(
                        client = self.async_client,
                        model = self.model_name,
                        messages = messages,
                        temperature=temperature,
                        **kwargs
                    ):
                if chunk is not None:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return
                    
    
    def __call__(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, response_format: Optional[Dict] = None, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Union[str, Dict[str, Any]]]:
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        if tools is not None and len(tools) > 0:
            return self.tool_calling(
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )

        start = time.time()
        try:

            content = _openai_text_completion(
                client = self.client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                response_format = response_format,
                **kwargs
            )
                
        except Exception as e:

            if not self.multimodal:
                logger.warning("Switching to multimodal")
                self.multimodal = True
                return self(messages, temperature, response_format, tools, **kwargs)

            else:
                # Handle edge cases
                logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
                return None

        end = time.time()
        logger.info(f"Completion time of {self.model_name}: {end - start}s")
        
        return content
    
    async def ainvoke(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, response_format: Optional[Dict] = None, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Union[str, Dict[str, Any]]]:
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)

        if tools is not None and len(tools) > 0:
            return await self.tool_calling_async(
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )

        start = time.time()
        try:

            content = await _openai_text_completion_async(
                client = self.async_client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                response_format = response_format,
                **kwargs
            )
                
        except Exception as e:

            if not self.multimodal:
                logger.warning("Switching to multimodal")
                self.multimodal = True
                return self(messages, temperature, response_format, tools, **kwargs)

            else:
                # Handle edge cases
                logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
                return None

        end = time.time()
        logger.info(f"Completion time of {self.model_name}: {end - start}s")
        
        return content
    
    def tool_calling(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)
            
        try:
            start = time.time()
            tool_call_result = _openai_tool_calling(
                client = self.client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )
            
            end = time.time()
            logger.info(f"Completion time of {self.model_name}: {end - start}s")

            return tool_call_result
        except Exception as e:
            # Handle edge cases
            logger.error(f"Error with API Key ending {str(self._api_key)[-5:]} : {e}")
            return None
        
    async def tool_calling_async(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)
            
        try:
            start = time.time()
            tool_call_result = await _openai_tool_calling_async(
                client = self.async_client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )

            print(tool_call_result)
            
            end = time.time()
            logger.info(f"Completion time of {self.model_name}: {end - start}s")

            return tool_call_result
        except Exception as e:
            # Handle edge cases
            api_key_suffix = getattr(self, '_OpenAIWrapper_api_key', getattr(self, '_ChatGPT_api_key', 'unknown'))
            logger.error(f"Error with API Key ending {str(api_key_suffix)[-5:]} : {e}")
            return None


    def stream_tool_calling(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Iterator[Optional[Dict[str, Any]]]:
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)    

        try:
            completion = _openai_text_completion_stream(
                client = self.client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )

            function_sample = {
                'id': None,
                'function': {
                    'name': None,
                    'arguments': ''
                },
                'type': 'function'
            }
            # Fix both content and function
            if completion:
                current_type = None
                current_func = None
                for chunk in completion:
                    if chunk is None:
                        continue
                    if chunk.choices[0].finish_reason is not None:
                        if current_func is not None:
                            current_func['function']['arguments'] = json.loads(current_func['function']['arguments'])
                            yield current_func
                            current_func = None
                        continue
                    
                    if chunk.choices[0].delta.tool_calls is not None:
                        print('Move to tool call')
                        
                        if current_type != 'tool_call':
                            current_type = 'tool_call'
                            
                            current_func = deepcopy(function_sample)
                            current_func['id'] = chunk.choices[0].delta.tool_calls[-1].id
                            current_func['function']['name'] = chunk.choices[0].delta.tool_calls[-1].function.name

                        else:
                            if chunk.choices[0].delta.tool_calls[-1].function.name is not None:
                                # New function call started - yield previous one
                                if current_func is not None:
                                    current_func['function']['arguments'] = json.loads(current_func['function']['arguments'])
                                    yield current_func
                                
                                current_func = deepcopy(function_sample)
                                current_func['id'] = chunk.choices[0].delta.tool_calls[-1].id
                                current_func['function']['name'] = chunk.choices[0].delta.tool_calls[-1].function.name
                            else:
                                # Continue building current function arguments
                                if current_func is not None:
                                    current_func['function']['arguments'] += chunk.choices[0].delta.tool_calls[-1].function.arguments

                    elif chunk.choices[0].delta.content is not None:
                        # Switch to content mode
                        if current_func is not None:
                            current_func['function']['arguments'] = json.loads(current_func['function']['arguments'])
                            yield current_func
                            current_func = None
                        
                        if current_type != 'content':
                            current_type = 'content'
                        
                        yield {'type': 'content', 'content': chunk.choices[0].delta.content}
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return None
        
    async def stream_tool_calling_async(self, messages: List[Dict[str, Union[str, List[Dict]]]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> AsyncIterator[Optional[Dict[str, Any]]]:
        if not self.system:
            messages = convert_non_system_prompts(messages)

        if self.multimodal:
            messages = convert_to_multimodal_format(messages)    

        try:
            completion = _openai_text_completion_stream_async(
                client = self.async_client,
                model = self.model_name,
                messages = messages,
                temperature=temperature,
                tools = tools,
                **kwargs
            )

            function_sample = {
                'id': None,
                'function': {
                    'name': None,
                    'arguments': ''
                },
                'type': 'function'
            }

            if completion:
                current_type = None
                current_func = None
                async for chunk in completion:
                    if chunk is None:
                        continue
                    if chunk.choices[0].finish_reason is not None:
                        if current_func is not None:
                            yield current_func
                            current_func = None
                        continue
                    
                    if chunk.choices[0].delta.tool_calls is not None:

                        if current_type != 'tool_call':
                            current_type = 'tool_call'
                            
                            current_func = deepcopy(function_sample)
                            current_func['id'] = chunk.choices[0].delta.tool_calls[-1].id
                            current_func['function']['name'] = chunk.choices[0].delta.tool_calls[-1].function.name

                        else:
                            if chunk.choices[0].delta.tool_calls[-1].function.name is not None:
                                # New function call started - yield previous one
                                if current_func is not None:
                                    yield current_func
                                
                                current_func = deepcopy(function_sample)
                                current_func['id'] = chunk.choices[0].delta.tool_calls[-1].id
                                current_func['function']['name'] = chunk.choices[0].delta.tool_calls[-1].function.name
                            else:
                                # Continue building current function arguments
                                if current_func is not None:
                                    current_func['function']['arguments'] += chunk.choices[0].delta.tool_calls[-1].function.arguments
                    
                    elif chunk.choices[0].delta.content is not None:
                        # Switch to content mode
                        if current_func is not None:
                            yield current_func
                            current_func = None
                        
                        if current_type != 'content':
                            current_type = 'content'
                        
                        yield {'type': 'content', 'content': chunk.choices[0].delta.content}
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return


class ChatGPT(OpenAIWrapper):
    def __init__(self, model_name: str = 'gpt-4.1-mini', engine: str = 'davinci-codex', max_tokens: int = 16384, api_key: Optional[str] = None,  **kwargs: Any) -> None:
        self.__initialize_client(model_name, engine, max_tokens, api_key, **kwargs)
        
    def __initialize_client(self, model_name: str = 'gpt-4.1-mini', engine: str = 'davinci-codex', max_tokens: int = 16384, api_key: Optional[str] = None, **kwargs: Any) -> None:
        self.model_name = model_name
        self.engine = engine

        if api_key is None:
            api_key=os.getenv('OPENAI_API_KEY')

        self._api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model_token = max_tokens
        self.max_tokens = min(self.model_token, max_tokens)
        self.multimodal = True
        self.system = True
        
    def stream(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Generator[str, None, None]:
        messages = convert_to_multimodal_format(messages)
        if 'gpt-5' in self.model_name:
            temperature = None
        return super().stream(messages, temperature, tools, **kwargs)
            
    async def astream(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any):
        messages = convert_to_multimodal_format(messages)
        if 'gpt-5' in self.model_name:
            temperature = None
        async for chunk in super().astream(messages, temperature, tools, **kwargs):
            yield chunk
            

    def __call__(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.4, response_format: Optional[Any] = None, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Union[Any, List[Any]]]:
        
        if 'gpt-5' in self.model_name:
            temperature = None
        return super().__call__(messages, temperature, response_format, tools, **kwargs)
            
    async def ainvoke(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.4, response_format: Optional[Any] = None, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Union[Any, List[Any]]]:
        
        if 'gpt-5' in self.model_name:
            temperature = None
        return await super().ainvoke(messages, temperature, response_format, tools, **kwargs)
            

    def tool_calling(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        
        if 'gpt-5' in self.model_name:
            temperature = None  
        return super().tool_calling(messages, temperature, tools, **kwargs)
    
    async def tool_calling_async(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        
        if 'gpt-5' in self.model_name:
            temperature = None 
        return await super().tool_calling_async(messages, temperature, tools, **kwargs)
    
    async def stream_tool_calling_async(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any):
        
        if 'gpt-5' in self.model_name:
            temperature = None 
        async for chunk in super().stream_tool_calling_async(messages, temperature, tools, **kwargs):
            yield chunk
    
    def stream_tool_calling(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> Iterator[Optional[Dict[str, Any]]]:
        
        if 'gpt-5' in self.model_name:
            temperature = None 
        return super().stream_tool_calling(messages, temperature, tools, **kwargs)
    
    
    def batch_call(self, list_messages: List[Any], key_list: List[str] = [], prefix: str = '', example_per_batch: int = 100, sleep_time: int = 10, sleep_step: int = 10) -> None:   
        
        """
        Batch call for ChatGPT
        list_messages: list of messages or custom batch
        transform: whether to transform the list of messages into batch
        prefix: prefix for the batch file
        example_per_batch: number of messages per batch
        """
        if not key_list or len(key_list) == 0:
            key_list = []
            for _ in list_messages:
                key_list.append(str(uuid4()))
        
        assert len(list_messages) == len(key_list), "Length of list_messages and key_list must be the same"

        list_messages = list_of_messages_to_batch_chatgpt(list_messages, key_list, example_per_batch=example_per_batch, model_type=self.model_name, prefix=prefix, max_tokens=self.max_tokens)

        if not os.path.exists('process'):
            os.mkdir('process')
        
        if not os.path.exists('batch'):
            os.mkdir('batch')
        
        for i, batch in enumerate(list_messages):
            
            # Save file to upload
            with open(f'process/process-{prefix}-{i}.jsonl', 'w', encoding='utf-8') as file:
                for message in batch:
                    json_line = json.dumps(message)
                    file.write(json_line + '\n')
            
            # Upload file
            if i % sleep_step == 0 and i != 0 and sleep_time != 0:
                logger.info(f"Sleeping for {sleep_time} seconds")
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
            logger.info(f"Batch {i} created")
            with open(f'batch/batch-{prefix}.jsonl', 'a', encoding='utf-8') as file:
                file.write(json.dumps({"id": batch_job.id}))
                file.write('\n')
    
    def recall_local_batch(self, prefix: str = '') -> List[Dict[str, Any]]:
        batch_ids = []
        if os.path.exists(f'batch/batch-{prefix}.jsonl'):
            with open(f'batch/batch-{prefix}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    batch_id = json.loads(line)['id']

                    batch = self.retrieve(batch_id)


                    logger.info(f"Batch {batch_id} status: {batch.status}")
                    batch_ids.append({
                        "id": batch_id,
                        "status": batch.status
                    })
        return batch_ids

    def recall_online_batch(self, prefix: str = '') -> List[Dict[str, Any]]:
        batch_ids = []
        batches = self.client.batches.list()
        for batch in batches:
                logger.info(f"Batch {batch.id} status: {batch.status}")
                batch_ids.append({
                    "id": batch.id,
                    "status": batch.status
                })
        return batch_ids

    def get_successful_messages(self, batch_ids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get successful messages from batch
        Batch_ids: list of batch ids (from recall_local_batch or recall_online_batch)
        """

        successful_messages = []
        for batch_id in batch_ids:
            batch = self.retrieve(batch_id["id"])
            if batch.status == 'completed':
                if batch.output_file_id:
                    file_response = self.client.files.content(batch.output_file_id).text
                    
                    # Save the response to a file
                    with open(f'batch/response-{batch_id["id"]}.jsonl', 'w', encoding='utf-8') as file:
                        file.write(file_response)
                    
                    with open(f'batch/response-{batch_id["id"]}.jsonl', 'r', encoding='utf-8') as file:
                        for line in file:
                            messages = json.loads(line)
                            messages_obj = {
                                "key": messages.get('custom_id'),
                                "text": messages.get('response').get('body').get('choices')[0].get('message').get('content')
                            }
                            successful_messages.append(messages_obj)

        return successful_messages

    def retrieve(self, batch_id: str) -> Any:
        return self.client.batches.retrieve(batch_id)
    

from collections import deque

class ClientOpenAIWrapper:
    def __init__(self, host: str, model_name: str, api_key: str, rpm: int, **kwargs: Any) -> None:
        self.llm = OpenAIWrapper(host = host, model_name=model_name, api_key=api_key, **kwargs)
        self.current_request = 0
        self.rpm = rpm
        self.request_time: deque = deque(maxlen=rpm)

    def check_max_rpm(self) -> bool:
        current_time = time.time() 
        last_1_min = current_time - 60
        if len(self.request_time) == self.rpm:
            begin_request = self.request_time[0]

            # Check if the first request is older than 1 min
            if begin_request <= last_1_min:
                while len(self.request_time) > 0 and self.request_time[0] <= last_1_min:
                    self.request_time.popleft()
                    if len(self.request_time) == 0:
                        self.request_time.append(current_time)
                        return False
                
                self.request_time.append(current_time)
                return False
            else:
                return True
        else:
            self.request_time.append(current_time)
            return False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.llm(*args, **kwargs)
    
    def stream(self,*args: Any, **kwargs: Any) -> Any:
        return self.llm.stream(*args, **kwargs)
    
    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        async for chunk in self.llm.astream(*args, **kwargs):
            yield chunk

    def __str__(self) -> str:
        return f"ClientOpenAIWarrper(model_name={self.llm.model_name}, rpm={self.rpm})"
    

class RotateOpenAIWrapper:


    def __init__(self, host: str, model_name: str, api_keys: Optional[List[str]] = None, api_prefix: Optional[str] = None, rpm: int = 10, **kwargs: Any) -> None:
        self._initialized = True
        self.model_name = model_name
        if not api_keys and api_prefix is not None:
            api_keys = get_all_api_key(api_prefix)
        self._api_keys = api_keys if api_keys is not None else []
        assert len(self._api_keys) > 0, "No api keys found"

        # Randomize the api_keys
        random.shuffle(self._api_keys)
        self.queue: deque = deque()
        for api_key in self._api_keys:
            self.queue.append(ClientOpenAIWrapper(host=host, model_name=model_name, api_key=api_key, rpm = rpm, **kwargs))

    def try_request(self, client: ClientOpenAIWrapper,  **kwargs: Any) -> Tuple[Any, bool]:
        try:
            print(client)
            return client( **kwargs), True
        except Exception as e:
            logger.error(f"Error: {e}")
            return '', False

    def __call__(self, messages: List[Dict[str, Union[str, List[Dict]]]], **kwargs: Any) -> Any:
        # Check if the first client in the queue has reached the maximum rpm
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client) # Add back to the queue
            client = self.queue.popleft() # Get the next client
            count += 1
            if count % len(self.queue) == 0:
                logger.warning("All clients have reached the maximum rpm. Wait for 30 seconds")
                time.sleep(30)
            if count > max_count:
                break
        
        # If the client has reached the maximum rpm, add back to the queue and try to make a request
        self.queue.append(client)

        response, success = self.try_request(client, messages = messages, **kwargs)
        
        tries = 0
        while not success:
            client = self.queue.popleft()

            # Check if the client has reached the maximum rpm
            if client.check_max_rpm():
                self.queue.append(client)
                continue

            # If the client has reached the maximum rpm, add back to the queue and try to make a request
            response, success = self.try_request(client, messages = messages, **kwargs)
            tries += 1
            time.sleep(min(tries, 10))
            self.queue.append(client)
            if tries > len(self.queue) * 2:
                logger.error("All clients have failed to make a request")
                return ''
            
        return response
        
    def stream(self, messages: List[Dict[str, Union[str, List[Dict]]]], **kwargs: Any) -> Generator[str, None, None]:
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client)
            client = self.queue.popleft()
            count += 1
            if count % len(self.queue) == 0:
                logger.warning("All clients have reached the maximum rpm. Wait for 10 seconds")
                time.sleep(10)
            if count > max_count:
                break

        self.queue.append(client)
        return client.stream(messages = messages, **kwargs)
    
    async def astream(self, messages: List[Dict[str, Union[str, List[Dict]]]], **kwargs: Any):
        client = self.queue.popleft()
        count = 0
        max_count = len(self.queue) * 2
        while client.check_max_rpm():
            self.queue.append(client)
            client = self.queue.popleft()
            count += 1
            if count % len(self.queue) == 0:
                logger.warning("All clients have reached the maximum rpm. Wait for 10 seconds")
                time.sleep(10)
            if count > max_count:
                break

        self.queue.append(client)
        async for chunk in client.astream(messages=messages, **kwargs):
            yield chunk
             
    def __repr__(self) -> str:
        return f"RoutingOpenAIWapper(model_name={self.model_name}, clients={len(self._api_keys)})"
    
    def __str__(self) -> str:
        return f"RoutingOpenAIWapper(model_name={self.model_name}, clients={len(self._api_keys)})"