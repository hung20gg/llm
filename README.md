# Library for using LLM easily
Support model directly from Hugging Face via `CoreLLMs` (with quantization), AWS Bedrock via `BedrockLLMs` and gemini via `Gemini`

Support function calling with `Gemini` and `ChatGPT` via `tools` and `**config` params

Support batch calling of `ChatGPT` (50% discount, why not)

Make sure to install requirements library: `boto3`, `transformers`, `openai` and `google-genai` (old: `google-generativeai`)

### Update:
**Gemini** is now switching to newer Python SDK version.

## Setup:

Create a `.env` file and pass every API key in it. Check each class for specific key.

- Gemini: `GEMINI_API_KEY` (new SDK) or `GENAI_API_KEY` (old SDK)
- GPT: `OPENAI_API_KEY`


## Usecase
Just call the model and it will work 

```python
from llm.llm.hugging_face import CoreLLMs

# Transformer model
llm = CoreLLMs(quantization = "int4") # if not passing model name, it will automatically use Llama 3

message = [
  {
    "role":"user",
    "content":"Hi there!"
  }
]

response = llm(message)
```

### For OpenAI wrapper (DeepSeek, Ollama, vLLM, etc.)

```python
from llm import OpenAIWrapper

host = "http://localhost:8000/v1" # vLLM host
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
api_key = None

llm = OpenAIWrapper(host = host, model_name = model_name, api_key = api_key)

```


### Rotate Key
For picking random key (rotate key to avoid max limit), set param `random_key = True`. Example:

```python
from llm import Gemini

llm = Gemini(random_key = True)
```

For auto try-catch with Queue (runnable but not fully optimized):

```python
from llm import RotateGemini # as Gemini

llm = RotateGemini(model_name = 'gemini-2.0-flash', rpm = 15)

```
Notice RPM (Request per minutes) for each model.
- Pro models: 2 RPM
- Flash models: 15 RPM
- Thinking: 10 RPM
- Lite: 30 RPM

Your API in `.env` for multiple key should be like this
```
GEMINI_API_KEY={key_0}
GEMINI_API_KEY_1={key_1}
GEMINI_API_KEY_2={key_2}
...
``` 