from llm.llm import BedRockLLMs
from llm.llm_utils import flatten_conversation

class LLMChat(BedRockLLMs):
    def __init__(self, model_name = "meta.llama3-8b-instruct-v1:0", 
                access_key = None,
                secret_key = None,
                secret_token = None,    
                region_name = "us-west-2",
                messages = []) -> None:
        super().__init__(model_name, access_key, secret_key, secret_token, region_name)
        self.messages = messages
        self.system_message = {"role":"system", 
                                "content":"""You are an helpful assistant for HR department. You should provide high quality and precise response to HR. If you are not sure about anything, feel free to ask for clarity. If you don't know about the answer, just response I don't know. The current recruitment date is May 2024"""}
        if len(messages) == 0:
            self.messages.append(self.system_message)

    def chat(self, message):
        if len(message) > 16:
            self.summarize()
        response = self(message)
        self.messages.append({"role":"user", "content":message})
        self.messages.append({"role":"assistant", "content":response})
        return response
    
    def summarize(self):
        summarize_message = {
            "role": "user",
            "content": f"Summarize the conversation in the chat below. Maintain key points and remove unnecessary information.\n{flatten_conversation(self.messages)}."
        }
        
        print("Summarize the conversation")
        summarize = self(summarize_message)
        system_prompt = self.system_message.copy()
        system_prompt["content"] += f"\n{summarize}"
        