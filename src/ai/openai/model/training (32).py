## \file /src/ai/openai/model/training (32).py
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
from types import SimpleNamespace
from typing import List, Dict
import pandas as pd
from openai import OpenAI

import header
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_loads_ns, j_dumps
from src.utils.csv import save_csv_file  
from src.utils.printer import pprint

class OpenAIModel:
    """OpenAI Model Class for interacting with the OpenAI API and managing the model."""

    model: str = "gpt-4o"
    client: OpenAI
    current_job_id: str
    assistant_id: str 
    assistant = None
    thread = None
    system_instruction: str
    dialogue_log_path: str | Path = gs.path.google_drive / 'AI' / f"{model}_{gs.now}.json"
    dialogue: List[Dict[str, str]] = []
    assistants: List[SimpleNamespace]
    models_list: List[str]

    def __init__(self, system_instruction: str = None, assistant_id: str = None):
        """Initialize the Model object with API key, assistant ID, and load available models and assistants.

        Args:
            system_instruction (str, optional): An optional system instruction for the model.
            assistant_id (str, optional): An optional assistant ID. Defaults to 'asst_dr5AgQnhhhnef5OSMzQ9zdk9'.
        """
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant_id = assistant_id or gs.credentials.openai.assistant_create_categories_with_description_from_product_titles
        self.system_instruction = system_instruction

        # Load assistant and thread during initialization
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()

    @property
    def list_models(self) -> List[str]:
        """Dynamically fetch and return available models from the OpenAI API.

        Returns:
            List[str]: A list of model IDs available via the OpenAI API.
        """
        try:
            models = self.client.models.list()
            model_list = [model['id'] for model in models['data']]
            logger.info(f"Loaded models: {model_list}")
            return model_list
        except Exception as ex:
            logger.error("An error occurred while loading models:", ex)
            return []

    @property
    def list_assistants(self) -> List[str]:
        """Dynamically load available assistants from a JSON file.

        Returns:
            List[str]: A list of assistant names.
        """
        try:
            self.assistants = j_loads_ns(gs.path.src / 'ai' / 'openai' / 'model' / 'assistants' / 'assistants.json')
            assistant_list = [assistant.name for assistant in self.assistants]
            logger.info(f"Loaded assistants: {assistant_list}")
            return assistant_list
        except Exception as ex:
            logger.error("An error occurred while loading assistants:", ex)
            return []

    def set_assistant(self, assistant_id: str):
        """Set the assistant using the provided assistant ID.

        Args:
            assistant_id (str): The ID of the assistant to set.
        """
        try:
            self.assistant_id = assistant_id
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
            logger.info(f"Assistant set successfully: {assistant_id}")
        except Exception as ex:
            logger.error("An error occurred while setting the assistant:", ex)

    def _save_dialogue(self):
        """Save the entire dialogue to the JSON file."""
        j_dumps(self.dialogue, self.dialogue_log_path)


    def determine_sentiment(self, message: str) -> str:
        """Determine the sentiment of a message (positive, negative, or neutral).

        Args:
            message (str): The message to analyze.

        Returns:
            str: The sentiment ('positive', 'negative', or 'neutral').
        """
        positive_words = ["good", "great", "excellent", "happy", "love", "wonderful", "amazing", "positive"]
        negative_words = ["bad", "terrible", "hate", "sad", "angry", "horrible", "negative", "awful"]
        neutral_words = ["okay", "fine", "neutral", "average", "moderate", "acceptable", "sufficient"]

        message_lower = message.lower()

        if any(word in message_lower for word in positive_words):
            return "positive"
        elif any(word in message_lower for word in negative_words):
            return "negative"
        elif any(word in message_lower for word in neutral_words):
            return "neutral"
        else:
            return "neutral"

    def ask(self, message: str, system_instruction: str = None, attempts: int = 3) -> str:
        """Send a message to the model and return the response, along with sentiment analysis.

        Args:
            message (str): The message to send to the model.
            system_instruction (str, optional): Optional system instruction.
            attempts (int, optional): Number of retry attempts. Defaults to 3.

        Returns:
            str: The response from the model.
        """
        try:
            # # Дообучебие - Загрузка предыдущих сообщений
            # messages = j_loads(gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.json')
            # if messages:
            #     # Отправка запроса к модели
            #     response = self.client.chat.completions.create(
            #         model=self.model,
            #         assistant=self.assistant_id,  # Учитываем assistant_id
            #         messages=messages,
            #         temperature=0,
            #     )
            #     reply = response.choices[0].message.content.strip()
                
            messages = []
            if self.system_instruction or system_instruction:
                system_instruction_escaped = (system_instruction or self.system_instruction).replace('"', r'\"')
                messages.append({"role": "system", "content": system_instruction_escaped})

            message_escaped = message.replace('"', r'\"')
            messages.append({"role": "user", "content": message_escaped})

            # Отправка запроса к модели
            response = self.client.chat.completions.create(
                model=self.model,
                assistant=self.assistant_id,  # Учитываем assistant_id
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip()

            self.client.chat.completions.create()
            # Анализ тональности
            sentiment = self.determine_sentiment(reply)

            # Добавление сообщений и тональности в диалог
            self.dialogue.append({"role": "system", "content": system_instruction or self.system_instruction})
            self.dialogue.append({"role": "user", "content": message_escaped})
            self.dialogue.append({"role": "assistant", "content": reply, "sentiment": sentiment})

            # Сохранение диалога
            self._save_dialogue()

            return reply
        except Exception as ex:
            logger.debug(f"An error occurred while sending the message: \n-----\n {pprint(messages)} \n-----\n", ex, True)
            time.sleep(3)  # Задержка перед повторной попыткой
            if attempts > 0:
                return self.ask(message, attempts - 1)
            return ""

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory.

        Args:
            data (str, optional): Path to a CSV file or CSV-formatted string with data.
            data_dir (Path | str, optional): Directory containing CSV files for training.
            data_file (Path | str, optional): Path to a single CSV file with training data.
            positive (bool, optional): Whether the data is positive or negative. Defaults to True.

        Returns:
            str | None: The job ID of the training job or None if an error occurred.
        """
        if not data_dir:
            data_dir = gs.path.google_drive / 'AI' / 'training'

        try:
            documents = j_loads(data if data else data_file if data_file else data_dir)

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
                j_dumps(job_ids_data, file)
            logger.info(f"Saved job ID {job_id} with description: {description}")
        except Exception as ex:
            logger.error(f"Error saving job ID {job_id}: {ex}")

def main(mode: str = 'train'):
    """Main function to execute the training and assistant management."""
    model = OpenAIModel()

    if mode == 'list_models':
        print("Available models:", model.list_models)
    elif mode == 'list_assistants':
        print("Available assistants:", model.list_assistants)
    else:
        response = model.ask("How can I help you today?")
        print(response)

if __name__ == "__main__":
    main('list_models')  # Can also use 'list_assistants' to list available assistants
