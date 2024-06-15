# Library for using LLM easily
Support model directly from Hugging Face via `CoreLLMs` and AWS Bedrock via `BedrockLLMs`

Make sure to install requirements library: `boto3`, `transformers`

## Usecase
Just call the model and it will work 
```
llm = CoreLLMs() # if not passing model name, it will automatically use Llama 3

message = [
  {
    "role":"user",
    "message":"Hi there!"
  }
]

response = llm(message)
```
