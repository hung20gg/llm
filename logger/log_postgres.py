import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

import psycopg2
import psycopg2.extras  # Add this import for Json functionality
import time
from uuid import uuid4
from PIL import Image

from llm.logger.log_base import LogBase

global_db_config = {
    'host': os.getenv('POSTGRE_HOST', 'localhost'),
    'database': os.getenv('POSTGRE_NAME', 'postgres'),
    'user': os.getenv('POSTGRE_USER', 'postgres'),
    'password': os.getenv('POSTGRE_PASSWORD', '12345678'),
    'port': os.getenv('POSTGRE_PORT', '5432')
}

class LLMLogPostgres(LogBase):

    def __init__(self, llm, db_config=None):
        """
        Initialize the logger with PostgreSQL database configuration.
        """
        self.model_name = llm.model_name
        if db_config is None:
            db_config = global_db_config
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        
        self.connect()
        self._initialize_db()
        
    def _initialize_db(self):
        """
        Initialize the PostgreSQL database and create the logs table if it doesn't exist.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages JSONB NOT NULL,
                    model_name VARCHAR(255),
                    run_name VARCHAR(255),
                    tag VARCHAR(255)
                )
            """)

            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_logs_timestamp ON llm_logs (timestamp);
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id SERIAL PRIMARY KEY,
                    log_id INTEGER REFERENCES llm_logs(id) ON DELETE CASCADE,
                    image_path VARCHAR(1024) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_images_log_id ON images (log_id);
            """)

            self.connection.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.connection.rollback()

    def connect(self):
        """
        Connect to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("Connected to PostgreSQL database")
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")


    def log(self, messages: list[dict], images_path: str|list[str],  run_name: str = '', tag: str = ''):
        """
        Log the messages and image path to the PostgreSQL database.
        
        :param image_path: Path to the image file.
        :param messages: List of messages to log.
        :param run_name: Name of the run.
        :param tag: Tag for the log entry.
        """
        if isinstance(images_path, str):
            images_path = [images_path]

        flag_image = len(images_path) > 0
        images = self.process_messages(messages, flag_image)
        if len(images) > 0 and len(images_path) == 0:
            images_path = images

        if not self.connection or self.connection.closed:
            self.connect()
            self._initialize_db()

        try:
            # Insert messages into llm_logs table
            self.cursor.execute("""
                INSERT INTO llm_logs (messages, model_name, run_name, tag)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (psycopg2.extras.Json(messages), self.model_name, run_name, tag))
            
            # Get the ID of the inserted log entry
            log_id = self.cursor.fetchone()[0]
            
            # Insert image paths into images table
            if images_path and len(images_path) > 0:
                for img_path in images_path:
                    self.cursor.execute("""
                        INSERT INTO images (log_id, image_path)
                        VALUES (%s, %s)
                    """, (log_id, img_path))
            self.connection.commit()
            print("Log entry added successfully")
        except Exception as e:
            print(f"Error logging to PostgreSQL database: {e}")
            self.connection.rollback()



if __name__ == "__main__":
    # Example usage
    from llm import ChatGPT
    llm = ChatGPT()  # Replace with your LLM instance
    logger = LLMLogPostgres(llm)

    messages = [
        {'role': 'user', 'content': 'Hello, how are you?'},
        {'role': 'assistant', 'content': 'I am fine, thank you!'}
    ]

    logger.log(image_path='../assets/spiderman.jpg', messages=messages, run_name='test_run', tag='test_tag')
    print("Log entry created successfully.")