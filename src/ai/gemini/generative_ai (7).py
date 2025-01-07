## \file /src/ai/gemini/generative_ai (7).py
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



""" Google generative ai """
from types import SimpleNamespace
import header  
import time
import base64
import argparse
from typing import Optional, List, Dict
from pathlib import Path
import textwrap
from datetime import datetime
import google.generativeai as genai
from src.logger.logger import logger

from src import gs
from src.utils.printer import pprint
from src.utils.csv import save_csv_file
from src.utils.jjson import j_dumps  
from src.utils.file import read_text_file, save_text_file, recursively_read_text_files
from src.utils.date_time import TimeoutCheck

timeout_check = TimeoutCheck()

class GoogleGenerativeAI:
    """GoogleGenerativeAI class for interacting with Google's Generative AI models."""

    model: genai.GenerativeModel
    dialogue_log_path: str | Path
    dialogue_txt_path: str | Path
    history_file: str | Path 
    system_instruction: str
    history_dir: str | Path  = gs.path.google_drive / 'AI' / 'history'
    history_file:str

    def __init__(self, api_key: str, system_instruction: Optional[str] = None, history_file: Optional[str | Path] = f'{gs.now}.txt', generation_config: dict = {"response_mime_type": "application/json"}):
        """Initialize GoogleGenerativeAI with the model and API key.

        Args:
            api_key (str): API key for Google Generative AI.
            system_instruction (Optional[str], optional): Optional system instruction for the model.
            history_file (Optional[str | Path], optional): Path to the history file.
            generation_config (dict): Configuration for the generation.
        """
        self.dialogue_log_path  = gs.path.google_drive / 'AI' / f"gemini_{gs.now}.json"
        self.dialogue_txt_path  = gs.path.google_drive / 'AI' / f"gemini_{gs.now}.txt"
         

        genai.configure(api_key=api_key)

        self.system_instruction = system_instruction
        models = [
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
        self.model = genai.GenerativeModel(
            models[2],
            generation_config=generation_config,
            system_instruction=system_instruction if system_instruction else None
        )
        self.history_file = self.history_dir / f'{gs.now}.txt' if not history_file else  self.history_dir / history_file


    def _save_dialogue(self, dialogue: list):
        """Save dialogue to a CSV file with a size limit and version tracking.

        If the content exceeds 10,000 characters, save the current file to disk 
        with a unique name. Continue appending to a new file and track history.

        Args:
            dialogue (list): Dialogue content to save.
        """
        # Read existing content from the dialogue file
        existing_content = read_text_file(self.dialogue_txt_path)
        new_content = existing_content + '\n'.join(dialogue) + '\n'

        # Check if content exceeds the size limit
        if len(new_content) > 10_000:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(self.dialogue_txt_path)
            new_file_name = f"{timestamp}_{base_name}"

            # Move the current content to the new file
            save_text_file(existing_content, new_file_name, mode='w')

            # Reset the content to only include the latest dialogue
            new_content = '\n'.join(dialogue) + '\n'

        # Save the updated content to the original file
        save_text_file(new_content, self.dialogue_txt_path, mode='w')


    def ask(self, q: str, system_instruction: Optional[str] = None, attempts: int = 3, total_wait_time: int = 0, no_log: bool = False, with_pretrain: bool = False, history_file: Optional[str | Path] = None ) -> Optional[str]:
        """Send a prompt to the model and return the response.

        Args:
            q (str): The prompt to send to the model.
            system_instruction (Optional[str], optional): Instruction for system role. Defaults to None.
            attempts (int, optional): Number of retry attempts in case of failure. Defaults to 3.
            total_wait_time (int, optional): The total time spent waiting between attempts. Defaults to 0.

        Returns:
            Optional[str]: The model's response or None if an error occurs.
        """

        self.history_file: Path = self.history_dir / history_file if history_file else self.history_file
        try:
            # Read conversation history
            history = read_text_file(self.history_file) if self.history_file else None

            # Prepare the complete prompt with history
            complete_prompt = f"{history}\n* question *\n{q}\n* answer **\n"

            # Prepare the content for the model
            content: dict = {
                "messages": [{"role": "user", "content": complete_prompt},
                             {"role": "system", "content": system_instruction} if system_instruction else None],
                "model": "gemini-1.5-flash-8b",
                "temperature": 0.7
            }

            messages = [{"role": "user", "content": complete_prompt},
                        {"role": "system", "content": system_instruction} if system_instruction else None]

            try:
                response = self.model.generate_content(str(messages))
            except Exception as ex:
                logger.debug("Ошибка ответа от модели\n", ex, True)
                return

            if not response:
                logger.debug("Не получил ответ от модели", None, True)
                return

            pprint(response.text, text_color='')

            if not no_log:
                self._save_dialogue([{"role": "system", "content": system_instruction} if system_instruction else self.system_instruction,
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": response}])
            
                self._save_dialogue(f"* question *\n{q}\n* answer **\n{response.text}\n")

            return response.text

        except Exception as ex:
            wait_time = 15  # Time to sleep in case of an error
            total_wait_time += wait_time
            logger.error(f"Error occurred", ex, False)

            if attempts > 0:
                return self.ask(q, system_instruction, attempts - 1, total_wait_time)
            else:
                logger.debug(f"Max attempts have been exceeded. Total wait time: {total_wait_time} seconds", None, False)
                attempts = 3
                return self.ask(q, system_instruction, attempts, total_wait_time)

    def describe_image(self, image_path: str, prompt: str = None) -> Optional[str]:
        """Describe an image using the text generation model.

        Args:
            image_path (str): Path to the image file.
            prompt (str, optional): Additional prompt for the model.

        Returns:
            Optional[str]: Description of the image or None in case of an error.
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        request = encoded_image

        try:
            # Send request to the model
            response = self.model.generate_content(request)
            return response.text
        except Exception as e:
            logger.error(f"Ошибка при описании изображения: {e}")
            return 

def chat():
    logger.debug("Привет, я ИИ ассистент компьюрного мастера Сергея Казаринова. Задавайте вопросы", None, False)
    print("Чтобы завершить чат, напишите 'exit'.\n")
    
    # Initialize the model with a system instruction if needed
    system_instruction = input("Введите системную инструкцию (или нажмите Enter, чтобы пропустить): @TODO: - сделать возможность чтения из .txt")
    ai = GoogleGenerativeAI(system_instruction=system_instruction if system_instruction else None)

    while True:
        # Get question from the user
        user_input = input("> вопрос\n> ")
        
        if user_input.lower() == 'exit':
            print("Чат завершен.")
            break
        
        # Send request to the model and get response
        response = ai.ask(q=user_input)
        
        # Print the response
        print(f">> ответ\n>> {response}\n")

if __name__ == "__main__":
    chat()
