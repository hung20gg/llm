class LogBase:
    
    def __init__(self, llm):
        self.llm = llm

    def log(self, messages: list[dict], image_path: str, run_name: str = '', tag: str = ''):

        """
        Log the messages and image path to the database.
        
        :param messages: List of messages to log.
        :param image_path: Path to the image file.
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __call__(self, messages: list[dict], image_path: str = '', run_name: str = '', tag: str = '', **kwargs):
        """
        Call the logger to log messages and image path.
        
        :param messages: List of messages to log.
        :param image_path: Path to the image file.
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        save_messages = messages.copy()
        response = self.llm(messages, **kwargs)

        save_messages.append({
            'role': 'assistant',
            'content': response
        })

        self.log(save_messages, image_path, run_name, tag)
        
        return response
    

    def stream(self, messages: list[dict], image_path: str = '', run_name: str = '', tag: str = '', **kwargs):
        """
        Stream the response from the LLM and log it.
        
        :param messages: List of messages to log.
        :param image_path: Path to the image file.
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        save_messages = messages.copy()
        response_stream = self.llm.stream(messages, **kwargs)
        response = ""

        for chunk in response_stream:
            if isinstance(chunk, str):
                response += chunk
                yield chunk

        save_messages.append({
            'role': 'assistant',
            'content': response
        })
        self.log(save_messages, image_path, run_name, tag)
