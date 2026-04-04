# Library for using LLMs easily
Support model wrappers for OpenAI, OpenAI-compatible hosts, Gemini, and Hugging Face local models.

Supports:
- text and multimodal chat
- function / tool calling for OpenAI and Gemini
- streaming responses and streaming tool-call events
- async APIs with `ainvoke` / `astream`
- batch calling for OpenAI wrappers
- key rotation via `random_key`

Make sure to install required libraries: `boto3`, `transformers`, `openai`, and `google-genai` (old: `google-generativeai`).

### Update:
**Gemini** is now switching to the newer Python SDK version.

## Setup

Create a `.env` file and pass every API key in it. Check each class for the specific key name.

- Gemini: `GEMINI_API_KEY` (new SDK) or `GENAI_API_KEY` (old SDK)
- OpenAI-compatible: `OPENAI_API_KEY`

## Quick start

```python
from llm import OpenAI

llm = OpenAI(model_name='gpt-4.1-mini')
messages = [
    {'role': 'user', 'content': 'Hi there!'}
]

response = llm(messages)
print(response)
```

## OpenAI-compatible hosts

Use `OpenAIWrapper` for non-OpenAI endpoints such as vLLM, DeepSeek, Ollama, or custom OpenAI-compatible servers.

```python
from llm import OpenAIWrapper

host = 'http://localhost:8000/v1'
model_name = 'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4'
llm = OpenAIWrapper(host=host, model_name=model_name, api_key=None)
```

## Gemini

Use `Gemini` for Google Gemini models with the new SDK. It supports streaming and tool calling via the same `tools` declaration style used by OpenAI.

```python
from llm import Gemini

llm = Gemini(model_name='gemini-2.0-flash')
```

## Tool calling

Tool calling is supported by OpenAI wrappers and Gemini. Pass a `tools` list when calling the model.

```python
from llm import OpenAI

llm = OpenAI(model_name='gpt-4.1-mini')
messages = [
    {'role': 'user', 'content': 'Get the current price of AAPL.'}
]
tools = [
    {
        'name': 'get_price',
        'description': 'Get the latest stock price for a symbol',
        'parameters': {
            'type': 'object',
            'properties': {
                'symbol': {'type': 'string'}
            },
            'required': ['symbol']
        }
    }
]

result = llm(messages, tools=tools)
print(result['content'])
print(result['tool_calls'])
```

### Tool call helpers

- `llm.tool_calling(messages, tools=tools)` returns a tool call result synchronously
- `await llm.tool_calling_async(messages, tools=tools)` returns the same result asynchronously
- `llm.stream_tool_calling(messages, tools=tools)` streams incremental tool-call events
- `await llm.stream_tool_calling_async(messages, tools=tools)` streams asynchronously

### Streaming tool-call events

`stream_tool_calling` yields dictionaries with two supported event types:
- `{'type': 'content', 'content': 'text chunk...'}`
- `{'type': 'function', 'id': ..., 'function': {'name': ..., 'arguments': '...'}}`

Example:

```python
for event in llm.stream_tool_calling(messages, tools=tools):
    if event['type'] == 'content':
        print('CHAT:', event['content'])
    else:
        print('TOOL CALL:', event['function']['name'], event['function']['arguments'])
```

## Streaming responses

Streaming is supported by `stream()` and `astream()` on most wrappers.

```python
for chunk in llm.stream(messages):
    print(chunk, end='')
```

Async streaming example:

```python
async for chunk in llm.astream(messages):
    print(chunk, end='')
```

Notes:
- `OpenAI` / `OpenAIWrapper` emit text chunks and may include reasoning markers like `<think>` / `</think>`.
- Reasoning content is automatically wrapped inside `<think>...</think>` blocks when streamed or returned.
- `Gemini` streams text chunks from the Gemini SDK.
- `HuggingFaceLLM` uses `TextIteratorStreamer` and yields tokens as they are generated.

## Image input

Image content can be passed as a path, `Image.Image` object, or base64 data URL.

```python
from llm import OpenAI

llm = OpenAI(model_name='gpt-4.1-mini')
messages = [
  {
    'role': 'user',
    'content': [
      {'type': 'text', 'text': 'Where is this?'},
      {'type': 'image', 'image_url': {'url': 'https://example.com/image.jpg'}}
    ]
  }
]
response = llm(messages)
```

## Logger

You can wrap any LLM instance with `LLMLogMongoDB` to automatically log usage and responses.

```python
from llm import OpenAI
from llm.logger.log_mongodb import LLMLogMongoDB

llm_ = OpenAI(model_name='gpt-4.1-mini')
llm = LLMLogMongoDB(llm_)
```

## Key rotation

Use `random_key=True` to select a random key from your environment variables:

```python
from llm import Gemini
llm = Gemini(random_key=True)
```

For rotating wrappers with retry queue logic, use `RotateGemini` or `RotateOpenAIWrapper`.

Example `.env` format for multiple keys:

```
GEMINI_API_KEY={key_0}
GEMINI_API_KEY_1={key_1}
GEMINI_API_KEY_2={key_2}
...
```
