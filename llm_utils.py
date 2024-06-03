import re
import json
import ast

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

def get_json_from_text_response(text_response):
    # Extract the JSON response from the text response
    text_responses = text_response.split("```")
    if len(text_responses) > 1:
        text_response = text_responses[1]
    else:
        text_response = text_responses[0]

        
    json_response = re.search(r"\[.*\]", text_response, re.DOTALL)
    if json_response:
        json_data = json_response.group(0)
        return check_json(json_data)
    json_response = re.search(r"\{.*\}", text_response, re.DOTALL)
    if json_response:
        json_data = json_response.group(0)
        return [check_json(json_data)]
    print("No JSON response found in text response")
    return text_response

def convert_format(data, has_system=True):
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
        new_messages.append({"role": role, "content": [{'type':'text', 'text':content}]})
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

def flatten_conversation(messages):
    conversation = []

    for message in messages:
        role = message['role']
        content = message['content']
        conversation.append(f"{role.capitalize()}: {content}")

    return "\n".join(conversation)

if __name__ == "__main__":

    messages = [
    {"role": "system", "content": "You are a friendly assistant"},
    {"role": "user", "content": "What is the weather today?"},
    {"role": "assistant", "content": "The weather today is sunny with a high of 75°F."},
    {"role": "user", "content": "Thank you!"},
    {"role": "assistant", "content": "You're welcome! Anything else you need?"}
]

    plain_text_conversation = flatten_conversation(messages)
    print(plain_text_conversation)