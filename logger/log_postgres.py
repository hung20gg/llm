import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

import psycopg2
import time

from llm.logger.log_base import LogBase

db_config = {
    'host': os.getenv('POSTGRE_HOST', 'localhost'),
    'database': os.getenv('POSTGRE_NAME', 'llm_logs'),
    'user': os.getenv('POSTGRE_USER', 'postgres'),
    'password': os.getenv('POSTGRE_PASSWORD', 'postgres'),
    'port': os.getenv('POSTGRE_PORT', '5432')
}

class LLMLogPostgres(LogBase):

    def __init__(self, db_config=None):
        """
        Initialize the logger with PostgreSQL database configuration.
        """
        if db_config is None:
            db_config = db_config
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self.connect()
        
    def _initialize_db(self):
        """
        Initialize the PostgreSQL database and create the logs table if it doesn't exist.
        """
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    request TEXT,
                    response TEXT,
                    model_name VARCHAR(255),
                    run_name VARCHAR(255),
                    tag VARCHAR(255)
                )
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