import json
import os 
import sys
import random
import time
from uuid import uuid4
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Dict, Any, Union, Iterator, Generator, Tuple, AsyncIterator
from copy import deepcopy
import re


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

def _get_reasoning_content(response):
    
    reasoning_content = getattr(response, 'reasoning_content', None)
    if not reasoning_content:
        reasoning_content = getattr(response, 'reasoning', None)
    
    return reasoning_content

def output_with_usage(response: Any, usage: Any, count_tokens: bool = False) -> Union[Dict[str, Any], Any]:
    if count_tokens:
        return {
            "response": response,
            "input_token": usage.prompt_tokens,
            "output_token": usage.completion_tokens,
            "total_token": usage.total_tokens
        }
    return response

_THINK_TAG_RE = re.compile(r"</?think>")

def format_reasoning_content(
    reasoning_content: Optional[str],
    content: Optional[str]
) -> Optional[str]:
    # Không có reasoning_content thì trả content như cũ
    if reasoning_content is None:
        return content

    reasoning_content = reasoning_content.strip()
    if not reasoning_content:
        return content

    # Bỏ tag <think> nếu input đã có sẵn
    cleaned = _THINK_TAG_RE.sub("", reasoning_content).strip()

    # Nếu sau khi bỏ tag mà trống luôn thì cũng coi như không có reasoning
    if not cleaned:
        return content

    wrapped = f"<think>{cleaned}</think>"

    if content is None:
        return wrapped

    return f"{wrapped}{content}"
def _openai_text_completion_stream(client: OpenAI, **kwargs: Any) -> Iterator[Optional[Any]]:
    try:
        completion = client.chat.completions.create(
            stream=True,
            **kwargs
        )
        stream_iter = iter(completion)
        while True:
            try:
                chunk = next(stream_iter)
                yield chunk
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"Skipping invalid stream chunk: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in chat completion stream: {e}")
        yield None
        

async def _openai_text_completion_stream_async(client: AsyncOpenAI, **kwargs: Any) -> AsyncIterator[Optional[Any]]:
    try:
        completion = await client.chat.completions.create(
            stream=True,
            **kwargs
        )
        stream_iter = completion.__aiter__()
        while True:
            try:
                chunk = await stream_iter.__anext__()
                # print(chunk)
                yield chunk
            except StopAsyncIteration:
                break
            except Exception as e:
                logger.warning(f"Skipping invalid stream chunk: {e}")
                continue
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
    reasoning_content = _get_reasoning_content(response)
    return format_reasoning_content(reasoning_content, response.content)


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

    content = response.content
    reasoning_content = _get_reasoning_content(response)    
    return {
        "content": format_reasoning_content(reasoning_content, content),
        "reasoning_content": reasoning_content
    }


def _format_legacy_completion_choice(choice: Any, include_logprobs: bool = False) -> Union[str, Dict[str, Any]]:
    content = choice.text
    if not include_logprobs:
        return content

    lp = choice.logprobs
    token_logprobs = []
    if lp is not None:
        tokens = lp.tokens or []
        logprobs = lp.token_logprobs or []
        top_logprobs = lp.top_logprobs or []
        text_offsets = lp.text_offset or []
        for idx, token in enumerate(tokens):
            token_logprobs.append({
                "index": idx,
                "token": token,
                "logprob": logprobs[idx] if idx < len(logprobs) else None,
                "top_logprobs": top_logprobs[idx] if idx < len(top_logprobs) else None,
                "text_offset": text_offsets[idx] if idx < len(text_offsets) else None,
            })

    return {
        "content": content,
        "logprobs": token_logprobs,
    }


def _openai_legacy_completion(client: OpenAI, include_logprobs: bool = False, **kwargs: Any) -> Union[str, Dict[str, Any]]:
    completion = client.completions.create(
        **kwargs
    )
    logger.info(completion.usage)
    return _format_legacy_completion_choice(completion.choices[0], include_logprobs=include_logprobs)


async def _openai_legacy_completion_async(client: AsyncOpenAI, include_logprobs: bool = False, **kwargs: Any) -> Union[str, Dict[str, Any]]:
    completion = await client.completions.create(
        **kwargs
    )
    logger.info(completion.usage)
    return _format_legacy_completion_choice(completion.choices[0], include_logprobs=include_logprobs)


def _openai_tool_calling(client: OpenAI, **kwargs: Any) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            **kwargs
        )
    except Exception as e:
        logger.warning(f"Skipping invalid tool calling response: {e}")
        return {"content": "", "tool_calls": []}
    content = ""
    tool_calls = []
    response = completion.choices[0].message
    
    if response.tool_calls is not None:
        for choice in response.tool_calls:
            tool_calls.append(choice.model_dump())
    
    content = response.content
    reasoning_content = _get_reasoning_content(response)    

    return {
        "content": format_reasoning_content(reasoning_content, content),
        "tool_calls": tool_calls
    }
    
async def _openai_tool_calling_async(client: AsyncOpenAI, **kwargs: Any) -> Dict[str, Any]:
    try:
        completion = await client.chat.completions.create(
            **kwargs
        )
    except Exception as e:
        logger.warning(f"Skipping invalid tool calling response: {e}")
        return {"content": "", "tool_calls": []}
    content = ""
    tool_calls = []
    response = completion.choices[0].message
    
    if response.tool_calls is not None:
        for choice in response.tool_calls:
            tool_calls.append(choice.model_dump())
    
    content = response.content
    reasoning_content = _get_reasoning_content(response)

    return {
        "content": format_reasoning_content(reasoning_content, content),
        "tool_calls": tool_calls
    }


class OpenAIWrapper(LLM):
    def __init__(self, host: str, model_name: str, api_key: Optional[str] = None, api_prefix: Optional[str] = None, random_key: bool = False, multimodal: bool = False, ignore_quota: bool = True, system: bool = True, **kwargs: Any) -> None:
        self.__initiate_client(host, model_name, api_key, api_prefix, random_key, multimodal, ignore_quota, system, **kwargs)
        
        
    def __initiate_client(self, host: str, model_name: str, api_key: Optional[str] = None, api_prefix: Optional[str] = None, random_key: bool = False, multimodal: bool = False, ignore_quota: bool = True, system: bool = True, **kwargs: Any) -> None:
        self.host = host
        self.model_name = model_name
        self.api_key = api_key

        logger.info(f"Initializing OpenAIWrapper with model {model_name} at host {host} and API key {str(api_key)[-5:]}")

        if api_key is None and random_key:
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
        self._chat_template_tokenizers: Dict[str, Any] = {}
        self._chat_template_generation_configs: Dict[str, Any] = {}

    @staticmethod
    def _normalize_chat_template_model_name(model_name: str) -> str:
        if ":" in model_name:
            return model_name.split(":", 1)[1]
        return model_name

    @staticmethod
    def _normalize_eos_ids(eos_token_id: Any) -> List[int]:
        if eos_token_id is None:
            return []
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        return list(eos_token_id)

    @staticmethod
    def _decode_token_id(tokenizer: Any, token_id: int) -> str:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

    def _is_end_of_message_token(self, tokenizer: Any, token_id: int) -> bool:
        token_text = self._decode_token_id(tokenizer, token_id).strip()
        end_of_message_tokens = {
            "</s>",
            "<|end|>",
            "<|eot_id|>",
            "<|im_end|>",
            "<|endofmessage|>",
            "<|end_of_message|>",
            "<|end_of_turn|>",
            "<end_of_turn>",
            "<｜end▁of▁sentence｜>",
        }
        return token_text in end_of_message_tokens

    def register_chat_template(self, model_name: Optional[str] = None, **kwargs: Any) -> Any:
        source_model_name = model_name or self.model_name
        hf_model_name = self._normalize_chat_template_model_name(source_model_name)
        if hf_model_name in self._chat_template_tokenizers:
            logger.info(f"Using cached chat template tokenizer for {hf_model_name}")
            return self._chat_template_tokenizers[hf_model_name]

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required to apply a Hugging Face chat template. "
                "Install it or pass a prompt directly to completion()."
            ) from e

        logger.info(
            f"Registering chat template tokenizer for {hf_model_name} "
            f"from completion model {source_model_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, **kwargs)
        self._chat_template_tokenizers[hf_model_name] = tokenizer
        return tokenizer

    def register_generation_config(self, model_name: Optional[str] = None, **kwargs: Any) -> Optional[Any]:
        source_model_name = model_name or self.model_name
        hf_model_name = self._normalize_chat_template_model_name(source_model_name)
        if hf_model_name in self._chat_template_generation_configs:
            logger.info(f"Using cached generation config for {hf_model_name}")
            return self._chat_template_generation_configs[hf_model_name]

        try:
            from transformers import GenerationConfig
        except ImportError as e:
            raise ImportError(
                "transformers is required to load a Hugging Face generation config. "
                "Install it or pass a prompt directly to completion()."
            ) from e

        try:
            logger.info(
                f"Registering generation config for {hf_model_name} "
                f"from completion model {source_model_name}"
            )
            generation_config = GenerationConfig.from_pretrained(hf_model_name, **kwargs)
        except Exception as e:
            logger.warning(f"Could not load generation config for {hf_model_name}: {e}")
            generation_config = None

        self._chat_template_generation_configs[hf_model_name] = generation_config
        return generation_config

    def get_all_eos_text(
        self,
        tokenizer: Any,
        generation_config: Optional[Any] = None,
    ) -> List[str]:
        eos_texts: List[str] = []
        if generation_config is not None:
            eos_ids = self._normalize_eos_ids(getattr(generation_config, "eos_token_id", None))
            eos_texts.extend(
                self._decode_token_id(tokenizer, eos_id)
                for eos_id in eos_ids
            )

        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token and eos_token not in eos_texts:
            eos_texts.append(eos_token)

        return eos_texts

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        add_generation_prompt: Optional[bool] = None,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str:
        tokenizer = self.register_chat_template(model_name=model_name)
        if add_generation_prompt is None:
            prompt_without_generation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **kwargs,
            )
            token_ids = tokenizer(prompt_without_generation, add_special_tokens=False).get("input_ids", [])
            add_generation_prompt = bool(token_ids and self._is_end_of_message_token(tokenizer, token_ids[-1]))
            logger.info(
                f"Auto add_generation_prompt={add_generation_prompt} "
                f"for chat template model {self._normalize_chat_template_model_name(model_name or self.model_name)}"
            )

        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def _build_completion_prompt(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        chat_template_model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, List[str]]:
        if isinstance(prompt, list) and messages is None:
            messages = prompt
            prompt = None

        if prompt:
            if not isinstance(prompt, str):
                raise ValueError("completion() prompt must be a string. Pass chat messages via messages=... or as the first argument.")
            return prompt, []

        if messages is None:
            raise ValueError("Either prompt or messages must be provided for completion().")

        if not self.system:
            messages = convert_non_system_prompts(messages)

        tokenizer = self.register_chat_template(model_name=chat_template_model)
        generation_config = self.register_generation_config(model_name=chat_template_model)
        completion_prompt = self.apply_chat_template(
            messages,
            model_name=chat_template_model,
            **kwargs,
        )
        return completion_prompt, self.get_all_eos_text(tokenizer, generation_config)

    @staticmethod
    def _normalize_stop(stop: Optional[Union[str, List[str]]], eos_tokens: Optional[List[str]] = None) -> Optional[List[str]]:
        stop_tokens: List[str] = []
        if isinstance(stop, str):
            stop_tokens.append(stop)
        elif stop is not None:
            stop_tokens.extend(stop)

        for eos_token in eos_tokens or []:
            if eos_token and eos_token not in stop_tokens:
                stop_tokens.append(eos_token)

        return stop_tokens or None

    def completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_token: int = 12000,
        temperature: float = 0.6,
        chat_template_model: Optional[str] = None,
        add_generation_prompt: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: int = 0,
        **kwargs: Any,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        max_completion_token = kwargs.pop("max_competion_token", max_completion_token)
        stop = kwargs.pop("stop_token", stop)
        logprobs = kwargs.pop("logprods", logprobs)
        logprobs = max(0, min(int(logprobs or 0), 5))
        template_kwargs = kwargs.pop("chat_template_kwargs", {})
        if add_generation_prompt is not None:
            template_kwargs["add_generation_prompt"] = add_generation_prompt
        if reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort
        completion_prompt, eos_tokens = self._build_completion_prompt(
            prompt=prompt,
            messages=messages,
            chat_template_model=chat_template_model,
            **template_kwargs,
        )

        try:
            start = time.time()
            content = _openai_legacy_completion(
                client=self.client,
                include_logprobs=logprobs > 0,
                model=self.model_name,
                prompt=completion_prompt,
                stop=self._normalize_stop(stop, eos_tokens=eos_tokens),
                max_tokens=max_completion_token,
                temperature=temperature,
                **({"logprobs": logprobs} if logprobs > 0 else {}),
                **kwargs,
            )
            end = time.time()
            logger.info(f"Completion time of {self.model_name}: {end - start}s")
            return content
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return None

    async def acompletion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_token: int = 1,
        temperature: float = 0,
        chat_template_model: Optional[str] = None,
        add_generation_prompt: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        logprobs: int = 0,
        **kwargs: Any,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        max_completion_token = kwargs.pop("max_competion_token", max_completion_token)
        stop = kwargs.pop("stop_token", stop)
        logprobs = kwargs.pop("logprods", logprobs)
        logprobs = max(0, min(int(logprobs or 0), 5))
        template_kwargs = kwargs.pop("chat_template_kwargs", {})
        if add_generation_prompt is not None:
            template_kwargs["add_generation_prompt"] = add_generation_prompt
        if reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort
        completion_prompt, eos_tokens = self._build_completion_prompt(
            prompt=prompt,
            messages=messages,
            chat_template_model=chat_template_model,
            **template_kwargs,
        )

        try:
            start = time.time()
            content = await _openai_legacy_completion_async(
                client=self.async_client,
                include_logprobs=logprobs > 0,
                model=self.model_name,
                prompt=completion_prompt,
                stop=self._normalize_stop(stop, eos_tokens=eos_tokens),
                max_tokens=max_completion_token,
                temperature=temperature,
                **({"logprobs": logprobs} if logprobs > 0 else {}),
                **kwargs,
            )
            end = time.time()
            logger.info(f"Completion time of {self.model_name}: {end - start}s")
            return content
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return None

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
                is_thinking = False
                for chunk in completion:
                    if chunk is not None:
                        content = chunk.choices[0].delta.content                        
                        reasoning_content = _get_reasoning_content(chunk.choices[0].delta)

                        if reasoning_content is not None:
                            if not is_thinking:
                                is_thinking = True
                                yield '<think>'
                            yield reasoning_content
                        else:
                            if is_thinking:
                                is_thinking = False
                                yield '</think>'
                            if content:
                                yield content
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return None
        
    
    async def astream(self, messages: List[Dict[str, Union[str, List[Dict]]]], temperature: float = 0.6, tools: Optional[List[Any]] = None, **kwargs: Any) -> AsyncIterator[Optional[str]]:

        # Disable system prompt (for gemma or llama 3.2)
        if not self.system:
            messages = convert_non_system_prompts(messages)

        messages = convert_to_multimodal_format(messages)

        try:
            is_thinking = False
            async for chunk in _openai_text_completion_stream_async(
                        client = self.async_client,
                        model = self.model_name,
                        messages = messages,
                        temperature=temperature,
                        **kwargs
                    ):
                if chunk is not None:
                    
                    content = chunk.choices[0].delta.content
                    reasoning_content = _get_reasoning_content(chunk.choices[0].delta)

                    if reasoning_content is not None:
                        if not is_thinking:
                            is_thinking = True
                            yield '<think>'
                        yield reasoning_content
                    else:
                        if is_thinking:
                            is_thinking = False
                            yield '</think>'
                        if content:
                            yield content
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return
                    
    
    def __call__(self, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.6, response_format: Optional[Dict] = None, tools: Optional[List[Any]] = None, **kwargs: Any) -> Optional[Union[str, Dict[str, Any]]]:
        
        if not self.system:
            messages = convert_non_system_prompts(messages)

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
        
        return content['content'] if isinstance(content, dict) and 'content' in content else content
    
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

            function_sample: Dict[str, Any] = {
                'id': None,
                'function': {
                    'name': None,
                    'arguments': ''
                },
                'type': 'function'
            }
            # Fix both content and function
            def _parse_args(raw: str) -> str:
                if not raw:
                    return "{}"
                try:
                    json.loads(raw)  # validate
                    return raw
                except Exception:
                    return raw

            if completion:
                current_type: Optional[str] = None
                current_func: Optional[Dict[str, Any]] = None
                for chunk in completion:
                    if chunk is None:
                        if current_type == 'reasoning':
                            yield {'type': 'content', 'content': '</think>'}
                        continue
                    if chunk.choices[0].finish_reason is not None:
                        if current_type == 'reasoning':
                            yield {'type': 'content', 'content': '</think>'}
                        if current_func is not None:
                            current_func['function']['arguments'] = _parse_args(current_func['function']['arguments'])
                            yield current_func
                            current_func = None
                        continue

                    response = chunk.choices[0].delta

                    if _get_reasoning_content(response):
                        if current_func is not None:
                            yield current_func
                            current_func = None
                        
                        if current_type != 'reasoning':
                            current_type = 'reasoning'

                            yield {'type': 'content', 'content': '<think>'}
                        
                        yield {'type': 'content', 'content':  _get_reasoning_content(response)}

                    elif current_type == 'reasoning':
                        yield {'type': 'content', 'content': '</think>'}
                        
                    
                    
                    if response.tool_calls is not None:
                        
                        if current_type != 'tool_call':
                            current_type = 'tool_call'
                            
                            current_func = deepcopy(function_sample)
                            current_func['id'] = response.tool_calls[-1].id
                            current_func['function']['name'] = response.tool_calls[-1].function.name
                            initial_args = response.tool_calls[-1].function.arguments
                            if initial_args:
                                current_func['function']['arguments'] += initial_args

                        else:
                            if response.tool_calls[-1].function.name is not None:
                                # New function call started - yield previous one
                                if current_func is not None:
                                    current_func['function']['arguments'] = _parse_args(current_func['function']['arguments'])
                                    yield current_func
                                
                                current_func = deepcopy(function_sample)
                                current_func['id'] = response.tool_calls[-1].id
                                current_func['function']['name'] = response.tool_calls[-1].function.name
                                initial_args = response.tool_calls[-1].function.arguments
                                if initial_args:
                                    current_func['function']['arguments'] += initial_args
                            else:
                                # Continue building current function arguments
                                if current_func is not None:
                                    current_func['function']['arguments'] += response.tool_calls[-1].function.arguments


                    
                    elif response.content is not None:
                        # Switch to content mode
                        if current_func is not None:
                            current_func['function']['arguments'] = _parse_args(current_func['function']['arguments'])
                            yield current_func
                            current_func = None

                        if current_type != 'content':
                            current_type = 'content'
                        
                        yield {'type': 'content', 'content': response.content}
                    

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

            function_sample: Dict[str, Any] = {
                'id': None,
                'function': {
                    'name': None,
                    'arguments': ''
                },
                'type': 'function'
            }

            def _parse_args_async(raw: str) -> str:
                if not raw:
                    return "{}"
                try:
                    json.loads(raw)  # validate
                    return raw
                except Exception:
                    return raw

            if completion:
                current_type: Optional[str] = None
                current_func: Optional[Dict[str, Any]] = None
                async for chunk in completion:
                    if chunk is None:
                        if current_type == 'reasoning':
                            yield {'type': 'content', 'content': '</think>'}
                        continue
                    if chunk.choices[0].finish_reason is not None:
                        if current_type == 'reasoning':
                            yield {'type': 'content', 'content': '</think>'}
                        if current_func is not None:
                            current_func['function']['arguments'] = _parse_args_async(current_func['function']['arguments'])
                            yield current_func
                            current_func = None
                        continue

                    response = chunk.choices[0].delta


                    if _get_reasoning_content(response):
                        if current_func is not None:
                            yield current_func
                            current_func = None
                        
                        if current_type != 'reasoning':
                            current_type = 'reasoning'

                            yield {'type': 'content', 'content': '<think>'}
                        
                        yield {'type': 'content', 'content':  _get_reasoning_content(response)}
                        continue

                    elif current_type == 'reasoning':
                        yield {'type': 'content', 'content': '</think>'}
                        
                    
                    if response.tool_calls is not None:

                        if current_type != 'tool_call':
                            current_type = 'tool_call'
                            
                            current_func = deepcopy(function_sample)
                            current_func['id'] = response.tool_calls[-1].id
                            current_func['function']['name'] = response.tool_calls[-1].function.name
                            initial_args = response.tool_calls[-1].function.arguments
                            if initial_args:
                                current_func['function']['arguments'] += initial_args

                        else:
                            if response.tool_calls[-1].function.name is not None:
                                # New function call started - yield previous one
                                if current_func is not None:
                                    current_func['function']['arguments'] = _parse_args_async(current_func['function']['arguments'])
                                    yield current_func
                                
                                current_func = deepcopy(function_sample)
                                current_func['id'] = response.tool_calls[-1].id
                                current_func['function']['name'] = response.tool_calls[-1].function.name
                                initial_args = response.tool_calls[-1].function.arguments
                                if initial_args:
                                    current_func['function']['arguments'] += initial_args
                            else:
                                # Continue building current function arguments
                                if current_func is not None:
                                    current_func['function']['arguments'] += chunk.choices[0].delta.tool_calls[-1].function.arguments
                    
                    elif response.content is not None:
                        # Switch to content mode
                        if current_func is not None:
                            yield current_func
                            current_func = None

                        if current_type != 'content':
                            current_type = 'content'
                        
                        yield {'type': 'content', 'content': response.content}
        
        except Exception as e:
            logger.error(f"Error with API Key ending {self._api_key[-5:]} : {e}")
            return


class OpenAIGPT(OpenAIWrapper):
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
        self._chat_template_tokenizers = {}
        self._chat_template_generation_configs = {}
        
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
