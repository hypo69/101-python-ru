## \file /src/ai/gemini/generative_ai (4).py
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


import header  
import time
import base64
from typing import Optional, List, Dict
from pathlib import Path
import os
import pathlib
import textwrap
import google.generativeai as genai

from src.logger.logger import logger
from src import gs
from src.utils.printer import pprint
from src.utils.csv import save_csv_file
from src.utils.jjson import j_dumps  


class GoogleGenerativeAI:
    """GoogleGenerativeAI class for interacting with Google's Generative AI models."""

    model: genai.GenerativeModel
    dialogue_log_path: str | Path = gs.path.google_drive / 'AI' / f"gemini_{gs.now}.json"
    dialogue: List[Dict[str, str]] = []  # Список для хранения диалога
    system_instruction:str

    def __init__(self, system_instruction: Optional[str] = None, generation_config: dict = {"response_mime_type": "application/json"}):
        """Initialize GoogleGenerativeAI with the model and API key.

        Args:
            system_instruction (Optional[str], optional): Optional system instruction for the model.
            generation_config (dict): "response_mime_type": "text/html" | "text/plain" | "application/json" 
            "response_mime_type": 
        """
        genai.configure(api_key=gs.credentials.googleai.api_key)
        self.system_instruction = system_instruction
        # Using `response_mime_type` requires either a Gemini 1.5 Pro or 1.5 Flash model
        models = ["gemini-1.5-flash-8b-exp-0924","gemini-1.5-flash"]
        self.model = genai.GenerativeModel(
            models[0],
            generation_config = generation_config,
            system_instruction = system_instruction if system_instruction else None
        )

    def _save_dialogue(self):
        """Save the entire dialogue to a CSV file."""
        j_dumps(self.dialogue, self.dialogue_log_path)

    def ask(self, prompt: str, system_instruction:Optional[str] = None, attempts: int = 3) -> Optional[str]:
        """Send a prompt to the model and return the response.

        Args:
            prompt (str): The prompt to send to the model.
            attempts (int, optional): Number of retry attempts in case of failure. Defaults to 5.

        Returns:
            Optional[str]: The model's response or None if an error occurs.
        """
        try:
            # Send prompt to the model
            response = self.model.generate_content(prompt)
            reply = response.text

            # Add user prompt and model reply to the dialogue
            if system_instruction:
                self.dialogue.append({"role": "system", "content": self.system_instruction})

            self.dialogue.append({"role": "user", "content": prompt})
            self.dialogue.append({"role": "assistant", "content": reply})

            # Save the dialogue to a CSV file
            self._save_dialogue()

            return reply
        except Exception as ex:
            #logger.error(f"Generative AI prompt {pprint(prompt)}\n{attempts=}", ex, True)
            logger.error(f"Go sleep for 15 sec /n", ex, False)
            time.sleep(15)  # <- Generative AI rate limit: up to 3 requests per minute
           
            if attempts > 0:
                return self.ask(prompt, attempts - 1)
            else:
                logger.debug(f"Max attmpts have been exceeded", None, False)
                attempts = 3
                return self.ask(prompt, attempts)
            return 

    def describe_image(self, image_path: str, prompt: str = None) -> Optional[str]:
        """Описывает изображение с помощью модели генерации текста.

        Args:
            image_path (str): Путь к файлу изображения.
            prompt (str, optional): Дополнительный промпт для модели.

        Returns:
            Optional[str]: Описание изображения или None в случае ошибки.
        """
        ##################################################################################
        #
        #
        # Модель не определяет фото!
        # 
        # 
        # # Проверка поддержки модели для работы с изображениями
        # if self.model.generation_config.get("response_mime_type") != "application/json":
        #     logger.error("Текущая модель не поддерживает работу с изображениями.")
        #     return

        # Кодирование изображения в формат base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        request = encoded_image

        try:
            # Отправка запроса к модели
            response = self.model.generate_content(request)
            return response.text
        except Exception as e:
            logger.error(f"Ошибка при описании изображения: {e}")
            return 