## \file /src/ai/openai/model/training (31).py
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
from types import SimpleNamespace
from typing import List, Dict
import pandas as pd
from openai import OpenAI

import header
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_loads_ns, j_dumps
from src.utils.csv import save_csv_file  
from src.utils.convertors import csv2json_csv2dict
from src.utils.printer import pprint


class OpenAIModel:
    """OpenAI Model Class for interacting with the OpenAI API and managing the model."""

    models_list: list = ["gpt-3.5-turbo-instruct", "gpt-4o"]
    model: str = "gpt-4o"
    client: OpenAI
    current_job_id: str
    assistant_id: str 
    assistant = None
    thread = None
    system_instruction: str
    dialogue_log_path: str | Path = gs.path.google_drive / 'AI' / f"{model}_{gs.now}.json"
    dialogue: List[Dict[str, str]] = []
    assistants:list[SimpleNamespace] = j_loads_ns(gs.path.src / 'ai' / 'openai' / 'model' / 'assistants' / 'assistants.json')

    def __init__(self, system_instruction: str = None, assistant_id: str = None):
        """Initialize the Model object with API key and assistant ID from settings.

        Args:
            system_instruction (str, optional): An optional system instruction for the model.
            assistant_id (str, optional): An optional assistant ID. Defaults to 'asst_dr5AgQnhhhnef5OSMzQ9zdk9'.
        """
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant_id = assistant_id or gs.credentials.openai.assistant_create_categories_with_description_from_product_titles
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
            # Загрузка предыдущих сообщений
            messages = j_loads ( gs.path.google_drive / 'AI' / 'conversation' / 'dailogue.json')
            if messages:
                # Отправка запроса к модели
                response = self.client.chat.completions.create(
                    model=self.model,
                    assistant=self.assistant_id,  # Учитываем assistant_id
                    messages=messages,
                    temperature=0,
                )
                reply = response.choices[0].message.content.strip()
                
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

    def stream_w(self, data: list | dict | str | None = None, data_file_path: str | Path = None):
        """Send prepared chat data, either from a list/dictionary/string or from a file.
        Note:
            see `src.suppliers.chat_gpt.chat_gpt.py`

        Args:
            data (list | dict | str | None, optional): The chat data to send. Can be a list, dictionary, or a CSV-formatted string. Defaults to None.
            data_file_path (str | Path, optional): The path to a file containing chat data. Defaults to None.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the data provided is not of an expected format.
        """
    
        def get_incremented_filename(file_path: Path) -> Path:
            """Generate a new filename with an incremented number if the file already exists.

            Args:
                file_path (Path): The path to the file.

            Returns:
                Path: The new file path with an incremented number.
            """
            base_name = file_path.stem
            extension = file_path.suffix
            i = 1
            new_file_path = file_path

            while new_file_path.exists():
                new_file_name = f"{base_name}_{i}{extension}"
                new_file_path = file_path.with_name(new_file_name)
                i += 1

            return new_file_path

        if data is None and data_file_path is None:
            logger.error("No data provided")
            return

        if data_file_path:
            file_path = Path(data_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            new_file_path = get_incremented_filename(file_path)
        
            try:
                df_existing = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame(columns=["role", "content", "sentiment"])
            except Exception as ex:
                logger.error("Error loading CSV file: %s", ex)
                return

            messages = []
            current_message_length = 0
            last_user_message = None

            for _, row in df_existing.iterrows():
                role = row.get('role')
                content = row.get('content')
                sentiment = row.get('sentiment', 'neutral')

                if role and content:
                    message_dict = {
                        "role": role,
                        "content": content,
                        "sentiment": sentiment
                    }
                    if role == 'user':
                        last_user_message = message_dict

                    elif role == 'assistant':
                        last_assistant_message = message_dict

                        if last_user_message:
                            response = self.ask(last_user_message["content"])

                            interaction = [
                                {"role": "user", "content": last_user_message["content"], "sentiment": last_user_message["sentiment"]},
                                {"role": "assistant", "content": last_assistant_message["content"], "sentiment": last_assistant_message["sentiment"]},
                                {"role": "assistant", "content": response, "sentiment": "neutral"}
                            ]

                            df_interaction = pd.DataFrame(interaction)
                            try:
                                df_interaction.to_csv(new_file_path, mode='a', header=not new_file_path.exists(), index=False)
                                logger.info("Updated %s with new interaction data", new_file_path)
                            except Exception as ex:
                                logger.error("Error saving CSV file: %s", ex)

                            last_user_message = None
                            last_assistant_message = None

            if last_user_message and last_assistant_message:
                response = self.ask(last_user_message["content"])

                interaction = [
                    {"role": "user", "content": last_user_message["content"], "sentiment": last_user_message["sentiment"]},
                    {"role": "assistant", "content": last_assistant_message["content"], "sentiment": last_assistant_message["sentiment"]},
                    {"role": "assistant", "content": response, "sentiment": "neutral"}
                ]

                df_interaction = pd.DataFrame(interaction)
                try:
                    df_interaction.to_csv(new_file_path, mode='a', header=not new_file_path.exists(), index=False)
                    logger.info("Updated %s with remaining interaction data", new_file_path)
                except Exception as ex:
                    logger.error("Error saving CSV file: %s", ex)

        elif isinstance(data, (list, dict, str)):
            if isinstance(data, str):
                try:
                    data = pd.read_csv(pd.compat.StringIO(data))
                except Exception as ex:
                    logger.error("Error reading CSV data: %s", ex)
                    return

            if isinstance(data, pd.DataFrame):
                data.to_csv(get_incremented_filename(Path("chat_data.csv")), index=False)
                logger.info("Saved chat data to file")
            else:
                logger.error("Unsupported data type provided")
        else:
            raise ValueError("Invalid data type provided. Expected list, dict, str, or None.")


    def list_models(self) -> List[str]:
        """List all available models.

        Returns:
            List[str]: A list of model names.
        """
        try:
            response = self.client.models.list()
            models = [model['id'] for model in response['data']]
            return models
        except Exception as ex:
            logger.error("An error occurred while listing models:", ex)
            return []

def main(mode: str = 'train'):
    """Main function to execute the training and assistant management."""

    model = OpenAIModel()

    if mode == 'train':
        # Start training
        model.train()
    elif mode == 'list_models':
        # List available models
        models = model.list_models()
        print("Available models:", models)
    else:
        model.system_instruction = "You are a helpful assistant that provides concise and accurate information."
        response = model.ask("How can I help you today?")
        print(response)

if __name__ == "__main__":
    main('list_models')  # Change mode to 'list_models' to list available models
