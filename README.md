# Library for using LLM easily
Support model directly from Hugging Face via `CoreLLMs` (with quantization), AWS Bedrock via `BedrockLLMs` and gemini via `Gemini`

Make sure to install requirements library: `boto3`, `transformers`

## Setup:

Create a `.env` file and pass every API key in it. Check each class for specific key.

## Usecase
Just call the model and it will work 
```
llm = CoreLLMs() # if not passing model name, it will automatically use Llama 3

message = [
  {
    "role":"user",
    "content":"Hi there!"
  }
]

response = llm(message)
```
