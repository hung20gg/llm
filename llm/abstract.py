from pydantic import BaseModel

class LLM(BaseModel):
    
    input_token: int = 0
    output_token: int = 0
    
    def __call__(self, message, **kwargs) -> str:
        raise NotImplementedError()
    
    def reset_token(self):
        self.input_token = 0
        self.output_token = 0