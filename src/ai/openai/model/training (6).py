## \file /src/ai/openai/model/training (6).py
# -*- coding: utf-8 -*-

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


# Import necessary modules
import openai
from pathlib import Path
from typing import List, Dict
import header
from src import gs
from src.utils.jjson import j_dumps, j_loads, j_loads_ns
from src.logger.logger import logger
from openai import OpenAI

# client = OpenAI(
#   api_key=gs.credentials.openai.api_key
# )
class Model():
    """ OpenAI Model Class """
    client:OpenAI
    def __init__(self):
        """Initialize the Model object with API key and assistant ID from settings."""
        self.client = OpenAI(api_key=gs.credentials.openai.api_key)
        self.current_job_id = None
        self.assistant_id = gs.credentials.openai.assistant_id


    def predict(self, text: str) -> str:
        """
        Generate a response based on the input text.
        @param text: The input text for which to generate a response.
        @return: The generated response.
        """
        try:
            response = self.completions.create(
                engine="text-davinci-003",  # Or another engine as needed
                prompt=text,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].text.strip()
        except Exception as ex:
            print(f"Error generating response: {ex}")
            return "Sorry, I couldn't generate a response at this time."

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory.

        @param data: Training data as a string or path to a file.
        @param data_dir: Path to the directory containing training data files.
        @param data_file: Path to a file containing training data (if not using `data` or `data_dir`).
        @param positive: A boolean flag indicating if the data is positive.
        @return: Job ID of the training task or None if an error occurred.
        """
        ...
        try:
            if data:
                # If `data` is provided, load data from the string
                documents = j_loads(data)
            elif data_file:
                # If `data_file` is provided, load data from the file
                documents = j_loads(data_file)
            elif data_dir:
                # If `data_dir` is provided, read all files from the directory
                documents = []
                for file_path in Path(data_dir).glob('*.json'):
                    documents.extend(j_loads(file_path))
            else:
                raise ValueError("No data source provided")

            # Perform the training
            response = self.client.
            response = self.train(
                model="text-davinci-003",
                documents=documents,
                labels=["positive" if positive else "negative"] * len(documents),
                show_progress=True
            )
            self.current_job_id = response.id
            return response.id
            
        except Exception as ex:
            logger.error("An error occurred during training:", ex)
            return



    def save_job_id(self, job_id: str, description: str, filename: str = "job_ids.json"):
        """Save the job ID with description to a file.

        @param job_id: The job ID to be saved.
        @param description: Description of the training task.
        @param filename: Name of the file where job IDs are stored.

        Example:
            model.save_job_id(job_id, "Training job for assistant ID")
        """
        try:
            job_ids_file = Path(filename)
            if job_ids_file.exists():
                job_ids_data = j_loads(job_ids_file)
            else:
                job_ids_data = {}

            job_ids_data[job_id] = description

            j_dumps(job_ids_data, job_ids_file)
        except Exception as ex:
            logger.error("An error occurred while saving the job ID:", ex)

    def create_vector_store(self, client, name: str, file_paths: List[str] = None) -> str | None:
        """Create a vector store and optionally add files to it.

        @param client: API client instance for making requests.
        @param name: Name of the vector store to be created.
        @param file_paths: List of file paths to be added to the vector store.
        @return: ID of the created vector store or None if an error occurred.

        Example:
            vector_store_id = model.create_vector_store(client, "Product Documentation", file_paths)
        """
        try:
            if file_paths:
                file_ids = [self.upload_file(client, path) for path in file_paths]
                response = client.beta.vector_stores.create(
                    name=name,
                    file_ids=file_ids
                )
            else:
                response = client.beta.vector_stores.create(name=name)
            
            print(f"Vector store created: {response}")
            return response.id
        except Exception as ex:
            logger.error(f"An error occurred while creating the vector store: {ex}")
            return

    def upload_file(self, client, file_path: str) -> str:
        """Upload a file and return its ID.

        @param client: API client instance for making requests.
        @param file_path: Path of the file to be uploaded.
        @return: ID of the uploaded file or None if an error occurred.

        Example:
            file_id = model.upload_file(client, "path/to/file.pdf")
        """
        try:
            with open(file_path, "rb") as file:
                response = client.beta.files.create(file=file)
                file_id = response.id
                print(f"File uploaded: {file_id}")
                return file_id
        except Exception as ex:
            logger.error(f"An error occurred while uploading file {file_path}: {ex}")
            return

    def upload_files_to_vector_store(self, client, vector_store_id: str, file_paths: List[str]):
        """Upload files to an existing vector store.

        @param client: API client instance for making requests.
        @param vector_store_id: ID of the vector store where files will be added.
        @param file_paths: List of file paths to be uploaded.

        Example:
            model.upload_files_to_vector_store(client, vector_store_id, file_paths)
        """
        try:
            file_ids = [self.upload_file(client, path) for path in file_paths]
            response = client.beta.vector_stores.update(
                vector_store_id=vector_store_id,
                file_ids=file_ids
            )
            print(f"Files added to vector store: {response}")
        except Exception as ex:
            logger.error(f"An error occurred while uploading files: {ex}")

    def update_assistant_with_file_search(self, client, assistant_id: str, vector_store_id: str):
        """Update the assistant to include the file_search tool and link the vector store.

        @param client: API client instance for making requests.
        @param assistant_id: ID of the assistant to be updated.
        @param vector_store_id: ID of the vector store to be linked.

        Example:
            model.update_assistant_with_file_search(client, assistant_id, vector_store_id)
        """
        try:
            assistant = client.beta.assistants.get(assistant_id)
            updated_tools = assistant.get('tools', []) + [{"type": "file_search"}]
            updated_assistant = client.beta.assistants.update(
                assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            print(f"Assistant updated successfully: {updated_assistant}")
        except Exception as ex:
            logger.error(f"An error occurred while updating the assistant: {ex}")

def main():
    """Main function to execute the training and assistant management."""
    model = Model()
    #model.train()

if __name__ == "__main__":
    main()
