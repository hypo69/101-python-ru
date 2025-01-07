## \file /src/ai/openai/model/training (33).py
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




"""OpenAI Model Class for handling communication with the OpenAI API and training the model."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict
from openai import OpenAI

from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_loads_ns, j_dumps, pprint


class OpenAIModel:
    """Class for interacting with the OpenAI API and managing the model."""

    model: str = "gpt-4o"
    client: OpenAI
    assistant_id: str
    system_instruction: str
    dialogue_log_path: Path = gs.path.google_drive / 'AI' / f"{model}_{gs.now}.json"
    dialogue: List[Dict[str, str]] = []

    def __init__(self, system_instruction: str = None, assistant_id: str = None):
        """Initialize the model object with API key, assistant ID, and load available models and assistants."""
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.assistant_id = assistant_id or gs.credentials.openai.default_assistant_id
        self.system_instruction = system_instruction
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id) if assistant_id, else None
        self.thread = self.client.beta.threads.create()

    @property
    def list_models(self) -> List[str]:
        """Fetch available models from the OpenAI API."""
        try:
            models = self.client.models.list()
            return [model['id'] for model in models['data']]
        except Exception as ex:
            logger.error("Error loading models:", ex)
            return []

    @property
    def list_assistants(self) -> List[str]:
        """Load available assistants from a JSON file."""
        try:
            assistants = j_loads_ns(gs.path.src / 'ai' / 'openai' / 'model' / 'assistants' / 'assistants.json')
            return [assistant.name for assistant in assistants]
        except Exception as ex:
            logger.error("Error loading assistants:", ex)
            return []

    def set_assistant(self, assistant_id: str):
        """Set the assistant by the provided assistant ID."""
        try:
            self.assistant_id = assistant_id
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
            logger.info(f"Assistant set successfully: {assistant_id}")
        except Exception as ex:
            logger.error("Error setting assistant:", ex)

    def _save_dialogue(self):
        """Save the dialogue to a JSON file."""
        j_dumps(self.dialogue, self.dialogue_log_path)

    def ask(self, message: str, system_instruction: str = None, attempts: int = 3) -> str:
        """Send a message to the model and return the response."""
        try:
            messages = [{"role": "system", "content": system_instruction or self.system_instruction}] if self.system_instruction or system_instruction else []
            messages.append({"role": "user", "content": message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip()

            self.dialogue.extend([
                {"role": "system", "content": system_instruction or self.system_instruction},
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ])

            self._save_dialogue()
            return reply
        except Exception as ex:
            logger.debug(f"Error sending message: {pprint(messages)}", ex, True)
            time.sleep(3)
            return self.ask(message, attempts - 1) if attempts > 0 else ""

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
        data_dir = data_dir or gs.path.google_drive / 'AI' / 'training'

        try:
            documents = j_loads(data or data_file or data_dir)
            response = self.client.Training.create(
                model=self.model,
                documents=documents,
                labels=["positive" if positive else "negative"] * len(documents),
                show_progress=True,
            )
            return response.id
        except Exception as ex:
            logger.error("Training error:", ex)
            return

    def save_job_id(self, job_id: str, description: str, filename: str = "job_ids.json"):
        """Save the job ID with description to a file."""
        try:
            job_ids_data = j_loads(Path(filename).read_text()) if Path(filename).exists() else {}
            job_ids_data[job_id] = description
            Path(filename).write_text(j_dumps(job_ids_data))
            logger.info(f"Saved job ID {job_id} with description: {description}")
        except Exception as ex:
            logger.error(f"Error saving job ID {job_id}: {ex}")

def main(mode: str = 'train'):
    """Main function to execute training and assistant management."""
    model = OpenAIModel()

    if mode == 'list_models':
        print("Available models:", model.list_models)
    elif mode == 'list_assistants':
        print("Available assistants:", model.list_assistants)
    else:
        response = model.ask("How can I help you today?")
        print(response)

if __name__ == "__main__":
    main('list_models')
