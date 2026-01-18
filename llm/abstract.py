class LLM:
    
    input_token: int = 0
    output_token: int = 0

    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key

    
    def __call__(self, message, **kwargs) -> str:
        raise NotImplementedError()

    def invoke(self, message, **kwargs):
        return self.__call__(message, **kwargs)

    async def ainvoke(self, message, **kwargs):
        raise NotImplementedError()
    
    def stream(self, message, **kwargs):
        raise NotImplementedError()
    
    async def astream(self, message, **kwargs):
        raise NotImplementedError()
    
    def generate(self, message, **kwargs):
        return self.stream(message, **kwargs)

    async def agenerate(self, message, **kwargs):
        return await self.astream(message, **kwargs)
    
    def tool_calling(self, message, **kwargs):
        raise NotImplementedError()
    
    async def tool_calling_async(self, message, **kwargs):
        raise NotImplementedError()

    def batch_call(self, messages, **kwargs):
        raise NotImplementedError()
    
    def retrieve(self):
        raise NotImplementedError()
    
    def reset_token(self):
        self.input_token = 0
        self.output_token = 0
        
    def usage(self):
        return {
            'input_token': self.input_token,
            'output_token': self.output_token
        }