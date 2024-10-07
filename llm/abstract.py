class LLM:
    def __call__(self, message, **kwargs) -> str:
        raise NotImplementedError()