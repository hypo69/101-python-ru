## \file /src/ai/gemini/generative_ai (11).py
# -*- coding: utf-8 -*-
#! venv/Scripts/python.exe
#! venv/bin/python/python3.12

"""
.. module:: src.ai.gemini 
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
  
""" module: src.ai.gemini """


"""Google generative AI integration."""

import time
import json
from pathlib import Path
from datetime import datetime
import base64
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import google.generativeai as genai
from src.logger.logger import logger
from src import gs
from src.utils.printer import pprint
from src.utils.file import read_text_file, save_text_file
from src.utils.date_time import TimeoutCheck
from src.utils.jjson import j_loads, j_loads_ns, j_dumps

timeout_check = TimeoutCheck()

class GoogleGenerativeAI(BaseModel):
    """Class to interact with Google Generative AI models """
    model_config = {
        "arbitrary_types_allowed": True
    }
    api_key: str
    model_name: str = Field(default="gemini-1.5-flash-8b")
    history_file: Optional[Path] = None
    generation_config: Dict = Field(default_factory=lambda: {"response_mime_type": "text/plain"})
    mode: str = Field(default='debug')
    
    MODELS: List[str] = Field(default_factory=lambda: [
        "gemini-1.5-flash-8b",
        "gemini-2-13b",
        "gemini-3-20b"
    ])
    
    dialogue_log_path: Optional[Path] = None
    dialogue_txt_path: Optional[Path] = None
    history_dir: Optional[Path] = None
    history_txt_file: Optional[Path] = None
    history_json_file: Optional[Path] = None
    model: Optional[genai.GenerativeModel] = None

    def __init__(self, 
                 api_key: str, 
                 model_name: Optional[str] = None, 
                 generation_config: Optional[Dict] = None, 
                 system_instruction: Optional[str] = None, 
                 **kwargs):
        """Initialize GoogleGenerativeAI with additional settings."""
        
        # Paths initialization
        self.dialogue_log_path = gs.path.google_drive / 'AI' / 'log'
        self.dialogue_txt_path = self.dialogue_log_path / f"gemini_{gs.now}.txt"
        self.history_dir = gs.path.google_drive / 'AI' / 'history'
        self.history_txt_file = self.history_dir / f"gemini_{gs.now}.txt"
        self.history_json_file = self.history_dir / f"gemini_{gs.now}.json"

        # Обновляем kwargs с новыми параметрами
        kwargs.update({
            'api_key': api_key,
            'model_name': model_name or "gemini-1.5-flash-8b",
            'generation_config': generation_config or {"response_mime_type": "text/plain"},
            'system_instruction': system_instruction,
        })
        
        # Инициализируем Pydantic модель
        super().__init__(**kwargs)

        # Конфигурируем модель Generative AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name = model_name or self.model_name,
            generation_config = generation_config or self.generation_config,
            system_instruction = system_instruction
        )

    def _save_dialogue(self, dialogue: list):
        """Save dialogue to both txt and json files with size management.

        Args:
            dialogue (list): Dialogue content to save.
        """
        save_text_file(dialogue, self.history_txt_file, mode='+a')
        j_dumps(data=dialogue, file_path=self.history_json_file, mode='+a')

    def ask(self, q: str, history_file: str = None, attempts: int = 3) -> Optional[str]:
        """Send a prompt to the model and get the response.

        Args:
            q (str): The prompt to send.
            history_file (str, optional): History file to use. Defaults to None.
            attempts (int, optional): Number of retry attempts. Defaults to 3.

        Returns:
            Optional[str]: The model's response or None if an error occurs.
        """
        self.history_file = self.history_dir / history_file if history_file else self.history_txt_file

        try:
            messages = [{"role": "user", "content": q}]
            response = self.model.generate_content(q)

            if not response:
                logger.debug("No response from the model.")
                return None

            messages.append({"role": "assistant", "content": response.text})
            self._save_dialogue([messages])

            return response.text

        except Exception as ex:
            logger.error("Error during request", ex)
            if attempts > 0:
                time.sleep(15)
                return self.ask(q, history_file, attempts=attempts - 1)
            return None

    def describe_image(self, image_path: Path) -> Optional[str]:
        """Generate a description of an image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            Optional[str]: Description of the image or None if an error occurs.
        """
        try:
            with image_path.open('rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')

            response = self.model.generate_content(encoded_image)
            return response.text

        except Exception as ex:
            logger.error(f"Error describing image: {ex}")
            return None

def chat():
    """Run the interactive chat session."""
    logger.debug("Hello, I am the AI assistant. Ask your questions.")
    print("Type 'exit' to end the chat.\n")

    system_instruction = input("Enter system instruction (or press Enter to skip): ")
    ai = GoogleGenerativeAI(api_key="your_api_key", system_instruction=system_instruction or None)

    while True:
        user_input = input("> Question: ")
        if user_input.lower() == 'exit':
            print("Chat ended.")
            break

        response = ai.ask(q=user_input)
        print(f">> Response:\n{response}\n")

if __name__ == "__main__":
    chat()
