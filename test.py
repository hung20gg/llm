import sys
import os
from dotenv import load_dotenv
sys.path.append('..')

load_dotenv()

from llm import get_llm_wrapper


llm = get_llm_wrapper("groq:openai/gpt-oss-120b")

messages = [
    {"role": "user", "content": "Tell me about France"}
]

for chunk in llm.stream(messages):
    print(chunk, end="")


print(llm.tool_calling(messages, tools=[]))
