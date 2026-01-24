import threading
from uuid import uuid4
import os
from PIL import Image
import asyncio
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
class LogBase:
    
    def __init__(self, llm):
        self.llm = llm
        self.model_name = llm.model_name
        self.multimodal = llm.multimodal if hasattr(llm, 'multimodal') else False

    def process_messages(self, messages: list[dict], save_images: bool) -> list[dict]:
        """
        Process the messages to ensure they are in the correct format.
        
        :param messages: List of messages to process.
        :return: Processed list of messages.
        """
        images = []

        for message in messages:
            if isinstance(message['content'], list):

                text_content = ""
                for content in message['content']:
                    if content['type'] == 'text':
                        text_content += content['text'] + " "

                    if content['type'] in ['image', 'image_url']:
                        if not isinstance(content[content['type']], str):
                            if isinstance(content[content['type']], Image.Image):
                                if save_images:
                                    # Save the image to a file

                                    image_path = f"image_{uuid4()}.png"
                                    save_folder = os.path.join(current_dir, '..', '..', 'images')
                                    os.makedirs(save_folder, exist_ok=True)
                                    image_path = os.path.join(save_folder, image_path)

                                    content[content['type']].save(image_path)
                                    images.append(image_path)

                            else:
                                raise ValueError("Unsupported image data type. Expected PIL Image or file path string.")
                        else:
                            images.append(content[content['type']])
                if '<image>' not in text_content:
                    text_content = "<image>\n" + text_content
                message['content'] = text_content.strip()
        return images
        


    def log(self, messages: list[dict], images_path: str|list[str], run_name: str = '', tag: str = ''):

        """
        Log the messages and image path to the database.
        
        :param messages: List of messages to log.
        :param images_path: Path to the image file(s).
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __call__(self, messages: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
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

        logging_thread = threading.Thread(
            target=self.log,
            args=(save_messages, images_path, run_name, tag),
            daemon=True  # Allow program to exit without waiting for this thread
        )
        logging_thread.start()

        return response

    def tool_calling(self, messages: list[dict], tools: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
        """
        Call the LLM with tool calling and log the messages.
        
        :param messages: List of messages to log.
        :param tools: List of tools available for the LLM.
        :param images_path: Path to the image file(s).
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        save_messages = messages.copy()
        response = self.llm.tool_calling(messages, tools=tools, **kwargs)

        content = response.get('content')
        tool_calls = response.get('tool_calls')

        save_messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls
        })

        logging_thread = threading.Thread(
            target=self.log,
            args=(save_messages, images_path, run_name, tag),
            daemon=True  # Allow program to exit without waiting for this thread
        )
        logging_thread.start()

        return response

    def stream(self, messages: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
        """
        Stream the response from the LLM and log it.
        
        :param messages: List of messages to log.
        :param images_path: Path to the image file(s).
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

        logging_thread = threading.Thread(
            target=self.log,
            args=(save_messages, images_path, run_name, tag),
            daemon=True  # Allow program to exit without waiting for this thread
        )
        logging_thread.start()

    def astream(self, messages: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
        """
        Asynchronously stream the response from the LLM and log it.
        
        :param messages: List of messages to log.
        :param images_path: Path to the image file(s).
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        

        async def stream_and_log():
            save_messages = messages.copy()
            response_stream = self.llm.astream(messages, **kwargs)
            response = ""

            async for chunk in response_stream:
                if isinstance(chunk, str):
                    response += chunk
                    yield chunk

            save_messages.append({
                'role': 'assistant',
                'content': response
            })

            logging_thread = threading.Thread(
                target=self.log,
                args=(save_messages, images_path, run_name, tag),
                daemon=True  # Allow program to exit without waiting for this thread
            )
            logging_thread.start()

        return stream_and_log()


    async def tool_calling_async(self, messages: list[dict], tools: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
        """
        Call the LLM with tool calling asynchronously and log the messages.
        
        :param messages: List of messages to log.
        :param tools: List of tools available for the LLM.
        :param images_path: Path to the image file(s).
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        save_messages = messages.copy()
        response = await self.llm.tool_calling_async(messages, tools=tools, **kwargs)

        content = response.get('content')
        tool_calls = response.get('tool_calls')

        save_messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls
        })

        logging_thread = threading.Thread(
            target=self.log,
            args=(save_messages, images_path, run_name, tag),
            daemon=True  # Allow program to exit without waiting for this thread
        )
        logging_thread.start()

        return response
    
    def stream_tool_calling(self, messages: list[dict], tools: list[dict], images_path: str|list[str] = [], run_name: str = '', tag: str = '', **kwargs):
        """
        Stream the response from the LLM with tool calling and log it.
        
        :param messages: List of messages to log.
        :param tools: List of tools available for the LLM.
        :param images_path: Path to the image file(s).
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        save_messages = messages.copy()
        response_stream = self.llm.stream_tool_calling(messages, tools=tools, **kwargs)
        response = ""
        tool_calls = []


        for chunk in response_stream:
            if isinstance(chunk, dict):
                if chunk.get('type') == 'tool_call':
                    tool_calls.append(chunk)
                elif chunk.get('type') == 'text':
                    response += chunk.get('text', '')
                    
            yield chunk

        save_messages.append({
            'role': 'assistant',
            'content': response,
            'tool_calls': tool_calls
        })

        logging_thread = threading.Thread(
            target=self.log,
            args=(save_messages, images_path, run_name, tag),
            daemon=True  # Allow program to exit without waiting for this thread
        )
        logging_thread.start()