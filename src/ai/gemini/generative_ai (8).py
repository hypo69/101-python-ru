## \file /src/ai/gemini/generative_ai (8).py
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
from pathlib import Path
from datetime import datetime
import base64
import google.generativeai as genai
from src.logger.logger import logger
from src import gs
from src.utils.printer import pprint
from src.utils.file import read_text_file, save_text_file
from src.utils.date_time import TimeoutCheck

timeout_check = TimeoutCheck()

class GoogleGenerativeAI:
    """Class to interact with Google Generative AI models."""

    MODELS = [
        "gemini-1.5-flash-8b",
        "gemini-2-13b",
        "gemini-3-20b"
    ]

    def __init__(self, 
                 api_key: str, 
                 model_name: str = "gemini-1.5-flash-8b",
                 system_instruction: str = None, 
                 history_file: Path = None, 
                 generation_config: dict = {"response_mime_type": "text/plain"}):
        """Initialize GoogleGenerativeAI with the model and API key.

        Args:
            api_key (str): API key for Google Generative AI.
            model_name (str, optional): Name of the model to use. Defaults to "gemini-1.5-flash-8b".
            system_instruction (str, optional): Instruction for system role.
            history_file (Path, optional): Path to the history file.
            generation_config (dict, optional): Configuration for the generation.
                Defaults to {"response_mime_type": "text/plain"}.
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Invalid model name. Choose from {', '.join(self.MODELS)}")

        self.dialogue_log_path = gs.path.google_drive / 'AI' / f"gemini_{gs.now}.json"
        self.dialogue_txt_path = gs.path.google_drive / 'AI' / 'history'/ f"gemini_{gs.now}.txt"
        self.history_dir = gs.path.google_drive / 'AI' / 'history'
        self.history_txt_file = self.history_dir / history_file if history_file else self.history_dir / f'{gs.now}.txt' 
        self.history_json_file = self.history_dir / history_file if history_file else self.history_dir / f'{gs.now}.json'
        self.system_instruction = system_instruction

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )

    def _save_dialogue(self, dialogue: list):
        """Save dialogue to a file with size management.

        Args:
            dialogue (list): Dialogue content to save.
        """
        existing_content = read_text_file(self.history_file)
        new_content = f"{existing_content}\n{'\n'.join(dialogue)}\n"

        if len(new_content) > 100_000:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_file_path = self.history_file.parent / f"{timestamp}_{self.history_file.name}"
            save_text_file(existing_content, new_file_path, mode='w')
            new_content = '\n'.join(dialogue) + '\n'

        save_text_file(new_content, self.history_file, mode='w')

    def ask(self, 
        q: str,
        history_file:str = None,
        attempts: int = 3) -> str | None:
        """Send a prompt to the model and get the response.

        Args:
            q (str): The prompt to send.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 0.9.
            max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 256.
            stop (list[str], optional): List of stop sequences. Defaults to None.
            attempts (int, optional): Number of retry attempts.

        Returns:
            str | None: The model's response.
        """
        self.history_file = self.history_dir / history_file if history_file else self.history_file
        try:
            history = read_text_file(self.history_file)
            complete_prompt = f"{history}\n* question *\n{q}\n* answer **\n"

            messages = [
                {"role": "system", "content": self.system_instruction},  # Добавлено system_instruction
                {"role": "user", "content": complete_prompt}
            ]

            response = self.model.generate_content(
                str(messages), 
            )

            if not response:
                logger.debug("No response from the model.")
                return 

            #pprint(response.text)
            messages.append({"role":"assiatant","content":response.text})
            self._save_dialogue([messages])

            return response.text

        except Exception as ex:
            logger.error("Error during request", ex)
            if attempts > 0:
                time.sleep(15)
                return self.ask(q, attempts=attempts - 1)
            return 

    def describe_image(self, image_path: Path) -> str | None:
        """Generate a description of an image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            str | None: Description of the image.
        """
        try:
            with image_path.open('rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')

            response = self.model.generate_content(encoded_image)
            return response.text

        except Exception as ex:
            logger.error(f"Error describing image: {ex}")
            return

def chat():
    """Run the interactive chat session."""
    logger.debug("Hello, I am the AI assistant of Sergey Kazarinov. Ask your questions.")
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
