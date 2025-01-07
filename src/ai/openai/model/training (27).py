## \file /src/ai/openai/model/training (27).py
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




""" OpenAI Model Class for handling communication with the OpenAI API and training the model. """

import time
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

import header
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_dumps
from src.utils.csv import save_csv_file, read_csv_file  # Импортируем функции для работы с CSV
from src.utils.printer import pprint


class OpenAIModel:
    """OpenAI Model Class for interacting with the OpenAI API and managing the model."""

    models_list: list = ["gpt-3.5-turbo-instruct", "gpt-4o"]
    model: str = "gpt-4o"
    client: OpenAI
    current_job_id: str
    assistant = None
    thread = None
    system_instruction: str
    dialogue_log_path: str | Path = gs.path.google_drive / 'AI' / f"dialogue_{gs.now}.csv"
    dialogue: List[Dict[str, str]] = []

    def __init__(self, system_instruction: str = None):
        """Initialize the Model object with API key and assistant ID from settings.

        Args:
            system_instruction (str, optional): An optional system instruction for the model.
        """
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant = self.client.beta.assistants.retrieve(gs.credentials.openai.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.system_instruction = system_instruction

    def _save_dialogue(self):
        """Save the entire dialogue to the CSV file."""
        save_csv_file(self.dialogue, self.dialogue_log_path)

    def ask(self, message: str, system_instruction: str = None, attempts:int = 3) -> str:
        """Send a message to the model and return the response.

        Args:
            message (str): The message to send to the model.

        Returns:
            str: The response from the model.
        """
        try:
            # Form the list of messages, starting with system_instruction if provided
            messages = []
            if self.system_instruction or system_instruction:
                system_instruction_escaped = self.system_instruction.replace('"', r'\"') or system_instruction.replace('"', r'\"')
                messages.append({"role": "system", "content": system_instruction_escaped})

            # Add user message
            message_escaped = message.replace('"', r'\"')
            messages.append({"role": "user", "content": message_escaped})

            # Send request to API with the complete list of messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip()

            # Add messages to dialogue
            self.dialogue.append({"role": "system", "content": system_instruction if system_instruction else self.system_instruction})
            self.dialogue.append({"role": "user", "content": message_escaped})
            self.dialogue.append({"role": "assistant", "content": reply})

            # Save the dialogue
            self._save_dialogue()

            return reply
        except Exception as ex:
            logger.debug(f"An error occurred while sending the message: \n-----\n {pprint(messages)} \n-----\n", ex, True)
            time.sleep(3)  # <- 
            if attempts > 0:
                return self.ask(message, attempts - 1)
            return 


    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory.

        Args:
            data (str, optional): JSON-formatted string with data.
            data_dir (Path | str, optional): Directory containing JSON files for training.
            data_file (Path | str, optional): Single JSON file with training data.
            positive (bool, optional): Whether the data is positive or negative. Defaults to True.

        Returns:
            str | None: The job ID of the training job or None if an error occurred.
        """
        try:
            documents = []
            if data:
                documents = j_loads(data)
            elif data_file:
                documents = j_loads(data_file)
            elif data_dir:
                for file_path in Path(data_dir).glob('*.json'):
                    documents.extend(j_loads(file_path))
            else:
                raise ValueError("No data source provided")

            response = self.client.Training.create(
                model=self.model,
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

        Args:
            job_id (str): The job ID to save.
            description (str): Description of the job.
            filename (str, optional): The file to save job IDs. Defaults to "job_ids.json".
        """
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
        """Update the assistant to include the file_search tool and link the vector store.

        Args:
            assistant_id (str): The assistant ID to update.
            vector_store_id (str): The ID of the vector store to link.
        """
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
    model = OpenAIModel(system_instruction="You are a helpful assistant that provides concise and accurate information.")
    model.ask("How can I help you today?")

if __name__ == "__main__":
    main()
