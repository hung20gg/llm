import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

import pymongo
from datetime import datetime
import time

from llm.logger.log_base import LogBase

global_db_config = {
    'host': os.getenv('MONGO_HOST', 'localhost'),
    'port': int(os.getenv('MONGO_PORT', '27017')),
    'db_name': os.getenv('MONGO_DB', 'llm_logs'),
    'collection_name': os.getenv('MONGO_COLLECTION', 'logs'),
    'username': os.getenv('MONGO_USER', 'mongodb'),
    'password': os.getenv('MONGO_PASSWORD', '12345678'),
}

class LLMLogMongoDB(LogBase):

    def __init__(self, llm, db_config=None):
        """
        Initialize the logger with MongoDB configuration.
        """
        self.model_name = llm.model_name
        self.llm = llm
        if db_config is None:
            db_config = global_db_config
        self.db_config = db_config
        self.client = None
        self.db = None
        self.collection = None
        
        self.connect()
        self._initialize_db()
        
    def _initialize_db(self):
        """
        Initialize the MongoDB database and create indexes if needed.
        """
        if self.db is None:
            print("Database connection not established.")
            return
            
        try:
            # Create indexes for faster queries
            self.collection.create_index("timestamp")
            self.collection.create_index("model_name")
            self.collection.create_index("run_name")
            self.collection.create_index("tag")
            print("MongoDB collection initialized")
        except Exception as e:
            print(f"Error initializing database: {e}")

    def connect(self):
        """
        Connect to the MongoDB database.
        """
        try:
            connection_string = f"mongodb://{self.db_config['host']}:{self.db_config['port']}/"
            
            # Add authentication if provided
            if self.db_config['username'] and self.db_config['password']:
                connection_string = f"mongodb://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/"
                
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client[self.db_config['db_name']]
            self.collection = self.db[self.db_config['collection_name']]
            print("Connected to MongoDB database")
        except Exception as e:
            print(f"Error connecting to MongoDB database: {e}")

    def log(self, messages: list[dict], image_path: str, run_name: str = '', tag: str = ''):
        """
        Log the messages and image path to the MongoDB database.
        
        :param image_path: Path to the image file.
        :param messages: List of messages to log.
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        if not self.client:
            self.connect()
            self._initialize_db()

        try:
            document = {
                "timestamp": datetime.now(),
                "image_path": image_path,
                "messages": messages,
                "model_name": self.model_name,
                "run_name": run_name,
                "tag": tag
            }
            
            result = self.collection.insert_one(document)
            print(f"Log entry added successfully with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error logging to MongoDB database: {e}")


if __name__ == "__main__":
    # Example usage
    from llm import ChatGPT
    llm = ChatGPT()  # Replace with your LLM instance
    logger = LLMLogMongoDB(llm)

    messages = [
        {'role': 'user', 'content': 'Hello, how are you?'},
    ]
    
    generator = logger.stream(messages=messages, run_name='test_run', tag='test_tag')
    for chunk in generator:
        print(chunk, end='', flush=True)
    print(messages)
    print("Log entry created successfully.")