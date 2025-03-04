import re
import json
import ast
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import os

from PIL import Image
import base64
from io import BytesIO

def pil_to_base64(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Change format if needed
    return base64.b64encode(buffered.getvalue()).decode()

def img_to_pil(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return image


def get_all_api_key(key: str) -> list[str]:

    api_keys = []

    # Orginal API key
    api_key = os.getenv(key)
    if api_key:
        api_keys.append(api_key)
    
    count = 1
    while os.getenv(f"{key}_{count}"):
        api_keys.append(os.getenv(f"{key}_{count}"))
        count += 1
    
    print(f"Found {len(api_keys)} API keys")
    
    return api_keys



def list_of_messages_to_batch_chatgpt(messages, example_per_batch = 10000, prefix = '', model_type = 'gpt-4o-mini', max_tokens = 40000):
    list_of_batches = []
    for i in range(0, len(messages), example_per_batch):
        batch = messages[i:i+example_per_batch]
        batch_json = []
        for j, message in enumerate(batch):
            json_obj = {
                "custom_id": f"request_{prefix}_{i*example_per_batch+j}",
                "method": "POST",
                "url": '/v1/chat/completions',
                "body": {
                    "model": model_type,
                    "messages": message,
                    "max_tokens":max_tokens
                },
                
            }
            batch_json.append(json_obj)
        list_of_batches.append(batch_json)
    return list_of_batches



def check_json(json_data):
    try:
        return json.loads(json_data)
    except Exception as e1:
        try:
            print(e1)
            pattern = r'\\?:\s*false\\?'
            json_data = re.sub(pattern, ': False', json_data)
            
            pattern = r'\\?:\s*true\\?'
            json_data = re.sub(pattern, ': True', json_data)
            
            pattern = r'\\?:\s*null\\?'
            json_data = re.sub(pattern, ': None', json_data)
            
            # json_data = json_data.replace(": false,", ": False,").replace(": true,", ": True,").replace(": null,", ": None,")
            # json_data = json_data.replace(":false,", ": False,").replace(":true,", ": True,").replace(":null,", ": None,")
            json_data = ast.literal_eval(json_data)
            return json.dumps(json_data, indent=4)
        except Exception as e2:
            print(e2)
            print("Error in converting JSON response")
            return json_data

def get_json_from_text_response(text_response, new_method=False):
    # Extract the JSON response from the text response
    text_responses = text_response.split("```")
    if len(text_responses) > 1:
        text_response = text_responses[1]
    else:
        text_response = text_responses[0]

    if new_method:
        language = text_response.split("\n")[0]
        content = "\n".join(text_response.split("\n")[1:])
        return check_json(content)
        
    else:
        json_response = re.search(r"\{.*\}", text_response, re.DOTALL)
        if json_response:
            json_data = json_response.group(0)
            return [check_json(json_data)]   
        json_response = re.search(r"\[.*\]", text_response, re.DOTALL)
        if json_response:
            json_data = json_response.group(0)
            return check_json(json_data)

    print("No JSON response found in text response")
    return text_response



def get_code_from_text_response(text_response):
    # Extract the code response from the text response
    code_response = re.search(r"```.*```", text_response, re.DOTALL)
    if code_response:
        text_chunk = text_response.split("```")
        code_chunk = text_chunk[1::2]
        return_code = []
        for code in code_chunk:
            language = code.split("\n")[0]
            content = "\n".join(code.split("\n")[1:])
            return_code.append({'language': language, 'code': content})
        return return_code
    
    else:
        print("No code response found in text response")
        return [{'language': 'text', 'code': text_response}]



def convert_llama_format(data, has_system=True):
    prompt = ""
    sys_non_llama = ''
    for i, item in enumerate(data):
        if i == 0 and item["role"] == "system":
            if has_system:
                system_message = item["content"]
                prompt += f"<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n{system_message}<|eot_id|>\n"
            else:
                sys_non_llama = item["content"]+'\n'

        elif item["role"] == "user":
            if i==1 and not has_system:
                user_message = item["content"]
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{sys_non_llama}{user_message}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            else:
                user_message = item["content"]
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_message}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        elif item["role"] == "assistant":
            assistant_message = item["content"]
            prompt += f"{assistant_message}\n<|eot_id|>"

    return prompt.rstrip("\n")



def convert_to_multimodal_format(messages, has_system=True):
    new_messages = []
    if not has_system:
        messages = convert_non_system_prompts(messages)
    for i, item in enumerate(messages):
        role = item["role"]
        content = item["content"]
        if isinstance(content, str):
            new_messages.append({"role": role, "content": [{'type':'text', 'text':content}]})
        elif isinstance(content, list):
            new_content = []
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'image':
                    new_content.append(
                        {
                            'type': 'image_url',
                            'image_url': {'url': f"data:image/png;base64,{pil_to_base64(c['image'])}"}
                        }
                    )
                    
                else:
                    new_content.append(c)
            new_messages.append({"role": role, "content": new_content})
        else:
            new_messages.append({"role": role, "content": content})
               
    return new_messages



def convert_non_system_prompts(messages):
    new_messages = []
    if messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        for i in range(1,len(messages)):
            message = messages[i]['content']
            if i == 1:
                message = system_prompt + '\n' + message
            new_messages.append({"role": messages[i]['role'], "content": message})
        return new_messages
    return messages




def convert_to_gemini_format(messages, has_system=True):
    new_messages = []
    if not has_system:
        messages = convert_non_system_prompts(messages)
    for i, item in enumerate(messages):
        role = item["role"]
        content = item["content"]
        if role == "assistant":
            role = "model"
        if not isinstance(content, list):
            content = [content]
        new_messages.append({"role": role, "parts": content})
    return new_messages



def flatten_conversation(messages):
    conversation = []

    for message in messages:
        role = message['role']
        content = message['content']
        conversation.append(f"#### {role.capitalize()}\n\n: {content}")

    return "\n\n".join(conversation)

if __name__ == "__main__":

    messages = [
    {"role": "system", "content": "You are a friendly assistant"},
    {"role": "user", "content": "What is the weather today?"},
    {"role": "assistant", "content": "The weather today is sunny with a high of 75°F."},
    {"role": "user", "content": "Thank you!"},
    {"role": "assistant", "content": "You're welcome! Anything else you need?"}
]

    # plain_text_conversation = flatten_conversation(messages)
    # print(plain_text_conversation)
    
    code_sample = """
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

Here is another code snippet:
```javascript
const x = 10;
console.log(x);
```
    """
    
    json_sample = """
    ```json
    {
        "next_step": ["SELECT * FROM table_name WHERE condition == \\"text\\";"]
    }
    ```
    """
    
    print(get_json_from_text_response(json_sample))
    print(get_code_from_text_response(code_sample))