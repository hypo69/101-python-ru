## \file /src/ai/openai/model/training (8).py
# -*- coding: utf-8 -*-
#! venv/Scripts/python.exe
#! venv/bin/python/python3.12

"""
.. module:: src.ai.openai.model 
	:platform: Windows, Unix
	:synopsis:

"""


"""
	:platform: Windows, Unix
	:synopsis:

"""

"""
	:platform: Windows, Unix
	:synopsis:

"""

"""
  :platform: Windows, Unix

"""
"""
  :platform: Windows, Unix
  :platform: Windows, Unix
  :synopsis:
"""
  
""" module: src.ai.openai.model """



...

from urllib import response
import openai
from openai import OpenAI
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict
import header
from src import gs
from src.ai.openai.model.event_handler import EventHandler
from src.utils.jjson import j_dumps, j_loads
from src.utils.printer import pprint
from src.logger.logger import logger

class Model():
    """OpenAI Model Class"""
    
    client: OpenAI
    current_job_id = None
    assistant = None
    thread = None

    def __init__(self):
        """Initialize the Model object with API key and assistant ID from settings."""
        self.client = OpenAI(api_key = gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant = self.client.beta.assistants.retrieve(gs.credentials.openai.assistant_id)
        #self.assistant.construct()
        self.thread = self.client.beta.threads.create()


    
    
    def send_message(self,message:str):
        """Отрпавка сообщения в модель """
        message = self.client.completions.create(
          model="gpt-3.5-turbo-instruct",
          prompt=message,
          max_tokens=2400,
          temperature=0
        )
        return message.choices[0].text.strip()


    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
        try:
            # if data:
            #     documents = j_loads(data)
            # elif data_file:
            #     documents = j_loads(data_file)
            # elif data_dir:
            #     documents = []
            #     for file_path in Path(data_dir).glob('*.json'):
            #         documents.extend(j_loads(file_path))
            # else:
            #     raise ValueError("No data source provided")

            response = self.client.Training.create(
                model="text-davinci-003",
                documents=data,
                labels=["positive" if positive else "negative"] * len(documents),
                show_progress=True
            )
            self.current_job_id = response.id
            return response.id

        except Exception as ex:
            logger.error("An error occurred during training:", ex)
            return
        
    def save_job_id(self, job_id: str, description: str, filename: str = "job_ids.json"):
        """Save the job ID with description to a file."""
        try:
            job_ids_data = {}
            if Path(filename).exists():
                with open(filename, 'r') as file:
                    job_ids_data = j_loads(file.read())
            job_ids_data[job_id] = description
            with open(filename, 'w') as file:
                file.write(j_dumps(job_ids_data))
        except Exception as ex:
            logger.error("An error occurred while saving the job ID:", ex)

    def update_assistant_with_file_search(self, assistant_id: str, vector_store_id: str):
        """Update the assistant to include the file_search tool and link the vector store."""
        try:
            updated_assistant = self.client.Assistant.update(
                assistant_id,
                tool="file_search",
                file_search_vector_store=vector_store_id
            )
            logger.info(f"Assistant updated successfully: {updated_assistant}")
        except Exception as ex:
            logger.error("An error occurred while updating the assistant:", ex)

def main():
    """Main function to execute the training and assistant management."""
    model = Model()
    model.predict('Hello, world!')

if __name__ == "__main__":
    main()