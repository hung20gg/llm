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

def pil_to_base64(image, format="JPEG"):
    """
    Convert PIL image, file path, or numpy array to base64 string.
    Handles different image modes including RGBA, LA and P.
    
    Args:
        image: PIL Image, file path, or numpy array
        format: Output format (default: JPEG)
        
    Returns:
        Base64 encoded string of the image
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Handle format compatibility issues
    if format.upper() == "JPEG":
        # JPEG doesn't support transparency
        if image.mode == 'RGBA':
            # Create a white background and paste the image using alpha as mask
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode == 'LA':
            # Grayscale with alpha - use alpha as mask with white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[1])
            image = background
        elif image.mode == 'P':
            # Convert palette mode to RGB
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            # Convert any other modes to RGB
            image = image.convert('RGB')
    
    # Save to buffer
    buffered = BytesIO()
    image.save(buffered, format=format)
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

        if role == 'system':
            text_content = ''
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            # elif isinstance(content, list):
            #     for c in content:
            #         if c.get('type') == 'text':
            #             text_content += c['text'] + '\n'
            #
            new_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, str):
            new_messages.append({"role": role, "content": [{'type':'text', 'text':content}]})
        elif isinstance(content, list):
            new_content = []
            for c in content:
                if isinstance(c, dict) and c.get('type') in ('image', 'image_url'):
                    image_url = ""
                    # Add check if c['image'] is a image path or PIL image
                    if isinstance(c['image'], str):
                        if os.path.exists(c['image']):
                            image_url = f"data:image/jpeg;base64,{pil_to_base64(c['image'])}"
                        elif c['image'].startswith('data:image'):
                            image_url = c['image']
                            
                    elif isinstance(c['image'], Image.Image):
                        image_url = f"data:image/jpeg;base64,{pil_to_base64(c['image'])}"
                    
                    if image_url != "":        
                        new_content.append(
                            {
                                'type': 'image_url',
                                'image_url': {'url': f"data:image/jpeg;base64,{pil_to_base64(c['image'])}"}
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
                if isinstance(message, str):
                    message = system_prompt + '\n' + message
                elif isinstance(message, list):
                    for part in message:
                        if part.get('type') == 'text':
                            part['text'] = system_prompt + '\n' + part['text']
            
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
    
    code_sample = """
Okay, I understand. The user wants me to create a bar chart based on the provided radar chart data, with the title "Financial Sector Performance". Since the previous code is `None`, I will generate the code from scratch. I\'ll need to extract the data from the image to create the bar chart. From the image, I can approximate the values for each sector in Q1, Q2, Q3, and Q5. I\'ll use these approximate values to create the bar chart.\n\nHere\'s the Python code using Matplotlib to generate the bar chart:\n\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Data extracted from the radar chart image (approximate values)\nsectors = [\'Stock Market\', \'Real Estate\', \'Cryptocurrency\', \'Commodities\']\nq1 = [2000, 4000, 1000, 3000]\nq2 = [6000, 8000, 5000, 4000]\nq3 = [3000, 5000, 2000, 1000]\nq5 = [1000, 2000, 500, 1500]\n\n# Set the width of the bars\nbar_width = 0.2\n\n# Set the positions of the bars on the x-axis\nr1 = np.arange(len(sectors))\nr2 = [x + bar_width for x in r1]\nr3 = [x + bar_width for x in r2]\nr4 = [x + bar_width for x in r3]\n\n# Create the bar chart\nplt.figure(figsize=(10, 6))\nplt.bar(r1, q1, color=\'blue\', width=bar_width, edgecolor=\'grey\', label=\'Q1\')\nplt.bar(r2, q2, color=\'orange\', width=bar_width, edgecolor=\'grey\', label=\'Q2\')\nplt.bar(r3, q3, color=\'green\', width=bar_width, edgecolor=\'grey\', label=\'Q3\')\nplt.bar(r4, q5, color=\'red\', width=bar_width, edgecolor=\'grey\', label=\'Q5\')\n\n# Add labels and title
    """
    
    print(get_code_from_text_response(code_sample))