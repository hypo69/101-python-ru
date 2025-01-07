## \file /src/ai/openai/model/training (29).py
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




""" OpenAI Model Class for handling communication with the OpenAI API and training the model. """

import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
from openai import OpenAI

import header
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_dumps
from src.utils.csv import save_csv_file  
from src.utils.convertors import csv2json_csv2dict
from src.utils.printer import pprint


class OpenAIModel:
    """OpenAI Model Class for interacting with the OpenAI API and managing the model."""

    models_list: list = ["gpt-3.5-turbo-instruct", "gpt-4o"]
    model: str = "gpt-4o"
    client: OpenAI
    current_job_id: str
    assistant_id: str = 'asst_dr5AgQnhhhnef5OSMzQ9zdk9'  # Default assistant ID
    assistant = None
    thread = None
    system_instruction: str
    dialogue_log_path: str | Path = gs.path.google_drive / 'AI' / f"gpt-4o_{gs.now}.json"
    dialogue: List[Dict[str, str]] = []

    def __init__(self, system_instruction: str = None, assistant_id: str = None):
        """Initialize the Model object with API key and assistant ID from settings.

        Args:
            system_instruction (str, optional): An optional system instruction for the model.
            assistant_id (str, optional): An optional assistant ID. Defaults to 'asst_dr5AgQnhhhnef5OSMzQ9zdk9'.
        """
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant_id = self.client.beta.assistants.retrieve(assistant_id or gs.credentials.openai.assistant_id)
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.thread = self.client.beta.threads.create()
        self.system_instruction = system_instruction

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
        """Save the entire dialogue to the CSV file."""
        j_dumps(self.dialogue, self.dialogue_log_path)
        # save_csv_file() <- save to CSV if needed

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
            csv2json_csv2dict.csv2json(gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.csv', gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.json')
            # дообучение модели на последних диалогах
            messages = j_loads(gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.csv', gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.json')
            if messages:
                # Отправка запроса к модели
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
                reply = response.choices[0].message.content.strip()
                # Формирование сообщений для отправки
            messages = []
            if self.system_instruction or system_instruction:
                system_instruction_escaped = (system_instruction or self.system_instruction).replace('"', r'\"')
                messages.append({"role": "system", "content": system_instruction_escaped})

            message_escaped = message.replace('"', r'\"')
            messages.append({"role": "user", "content": message_escaped})

            # Отправка запроса к модели
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip()

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


def main(mode: str = 'train'):
    """Main function to execute the training and assistant management."""

    model = OpenAIModel()

    if mode == 'train':
        # Запуск тренировки
        model.train()

    model.system_instruction = "You are a helpful assistant that provides concise and accurate information."
    response = model.ask("How can I help you today?")
    print(response)

if __name__ == "__main__":
    main('ask')  # Запускаем с режимом 'ask' и использованием ассистента с указанным ID
