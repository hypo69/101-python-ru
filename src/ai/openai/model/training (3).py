## \file /src/ai/openai/model/training (3).py
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


""" Modile for trainig AI assistant """

...
from urllib import response
import openai
from openai import OpenAI
from pathlib import Path
from types import SimpleNamespace
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
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant = self.client.beta.assistants.retrieve(gs.credentials.openai.assistant_id)
        self.thread = self.client.beta.threads.create()

    def strem_w(self, data: list | dict | None = None, data_file: str | Path = None):
        """ отсылка приготовленых в грабере чатов 
        см `src.suppliers.chat_gpt.chat_gpt.py`
        """
        ...
        if not data and not data_file:
            logger.error("Ау! где данные?")
            return
        if data_file:
            data: dict = j_loads(data_file)
        
        counter = 0
        messages: list = []
        for index, item in enumerate(data):
            messages.extend(item)
            counter += 1
            if counter > 3:
                stream = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        print(chunk.choices[0].delta.content, end="")
                counter = 0
                messages = []
            ...

    def send_message(self, message: str):
        """Отрпавка сообщения в модель """
        try:
            message = self.client.completions.create(
                model="gpt-4o-mini",
                prompt=message,
                max_tokens=400,
                temperature=0
            )
            return message.choices[0].text.strip()
        except Exception as ex:
            logger.warning(ex, None, True)
            return str(ex)

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
        try:
            if data:
                documents = j_loads(data)
            elif data_file:
                documents = j_loads(data_file)
            elif data_dir:
                documents = []
                for file_path in Path(data_dir).glob('*.json'):
                    documents.extend(j_loads(file_path))
            else:
                raise ValueError("No data source provided")

            response = self.client.Training.create(
                model="text-davinci-003",
                documents=documents,
                labels=["positive" if positive else "negative"] * len(documents),
                show_progress=True
            )
            self.current_job_id = response.id
            return response.id

        except Exception as ex:
            logger.warning("An error occurred during training:", ex)
            return str(ex)

    def save_job_id(self, job_id: str, description: str, filename: str = "job_ids.json"):
        """Save the job ID with description to a file."""
        try:
            job_ids_file = Path(filename)
            if job_ids_file.exists():
                job_ids_data = j_loads(job_ids_file)
            else:
                job_ids_data = {}

            job_ids_data[job_id] = description

            j_dumps(job_ids_data, job_ids_file)
        except Exception as ex:
            logger.warning("An error occurred while saving the job ID:", ex)
            return str(ex)

    def create_vector_store(self, client, name: str, file_paths: list[str] = None) -> str | None:
        """Create a vector store and optionally add files to it."""
        try:
            if file_paths:
                file_ids = [self.upload_file(client, path) for path in file_paths]
                response = client.Beta.VectorStores.create(
                    name=name,
                    file_ids=file_ids
                )
            else:
                response = client.Beta.VectorStores.create(name=name)

            print(f"Vector store created: {response}")
            return response.id
        except Exception as ex:
            logger.warning(f"An error occurred while creating the vector store: {ex}")
            return str(ex)

    # def upload_file(self, client, file_path: str) -> str:
    #     """Upload a file and return its ID."""
    #     try:
    #         with open(file_path, "rb") as file:
    #             response = self.client. .Files.create(file=file)
    #             file_id = response.id
    #             print(f"File uploaded: {file_id}")
    #             return file_id
    #     except Exception as ex:
    #         logger.error(f"An error occurred while uploading file {file_path}: {ex}")
    #         return

    def upload_files_to_vector_store(self, client, vector_store_id: str, file_paths: list[str]):
        """Upload files to an existing vector store."""
        try:
            file_ids = [self.upload_file(client, path) for path in file_paths]
            response = client.Beta.VectorStores.update(
                id=vector_store_id,
                file_ids=file_ids
            )
            print(f"Files uploaded to vector store: {response}")
            return response
        except Exception as ex:
            logger.error(f"An error occurred while uploading files to vector store: {ex}")
            return str(ex)
