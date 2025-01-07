## \file /src/ai/openai/model/training (23).py
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




""" OpenAI Model Class for handling communication with the OpenAI API and training the model.  """

from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import header
import csv
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_dumps
from src.utils.printer import pprint
from openai import OpenAI


class OpenAIModel:
    """OpenAI Model Class"""
    models_list: list = ["gpt-3.5-turbo-instruct", "gpt-4o"]
    model: str = "gpt-4o"
    client: OpenAI
    current_job_id: str
    assistant = None
    thread = None
    system_instruction: str
    dialogue_log_path: str = "dialogue_log.csv"

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

    def _save_dialogue(self, role: str, content: str):
        """Save a dialogue message to the CSV file.

        Args:
            role (str): The role of the sender ('user' or 'assistant').
            content (str): The content of the message.
        """
        file_exists = Path(self.dialogue_log_path).exists()
        with open(self.dialogue_log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['role', 'content'])
            writer.writerow([role, content])

    def ask(self, message: str) -> str:
        """ Send a message to the model and return the response"""
        try:
            # Формируем список сообщений, начиная с system_instruction, если он указан
            messages = []
            if self.system_instruction:
                system_instruction_escaped = self.system_instruction.replace('"', r'\"')
                #messages.append(f'{{"role": "system", "content": "{system_instruction_escaped}"}}')
                messages.append({"role": "system", "content": system_instruction_escaped})
        
            # Добавляем сообщение пользователя
            message = message.replace('"', r'\"')
            #messages.append(f'{{"role": "user", "content": "{user}"}}')
            messages.append({"role": "user", "content": message})

            # Отправляем запрос к API с полным списком сообщений
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reply})
            ...

            # Сохраняем диалог
            self._save_dialogue('user', messages)
            self._save_dialogue('assistant', reply)

            return reply
        except Exception as ex:
            logger.debug(f"An error occurred while sending the message: \n-----\n {messages} \n-----\n", ex, None)
            return ""

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
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
    model = OpenAIModel(system_instruction="You are a helpful assistant that provides concise and accurate information.")
    model.stream_w(data_file_path=Path(gs.path.google_drive / 'chat_gpt' / 'conversation' / 'all_conversations.csv'))

if __name__ == "__main__":
    main()
