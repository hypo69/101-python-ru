## \file /src/ai/gemini/generative_ai (3).py
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

    def __init__(self, system_instruction: Optional[str] = None):
        """Initialize GoogleGenerativeAI with the model and API key.

        Args:
            system_instruction (Optional[str], optional): Optional system instruction for the model.
        """
        genai.configure(api_key=gs.credentials.googleai.api_key)
        self.system_instruction = system_instruction
        # Using `response_mime_type` requires either a Gemini 1.5 Pro or 1.5 Flash model
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={"response_mime_type": "application/json"},
            system_instruction=system_instruction if system_instruction else None
        )

    def _save_dialogue(self):
        """Save the entire dialogue to a CSV file."""
        j_dumps(self.dialogue, self.dialogue_log_path)

    def ask(self, prompt: str, attempts: int = 3) -> Optional[str]:
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
            self.dialogue.append({"role": "system", "content": self.system_instruction})
            self.dialogue.append({"role": "user", "content": prompt})
            self.dialogue.append({"role": "assistant", "content": reply})

            # Save the dialogue to a CSV file
            self._save_dialogue()

            return reply
        except Exception as ex:
            #logger.error(f"Generative AI prompt {pprint(prompt)}\n{attempts=}", ex, True)
            logger.debug(f"Go sleep for 15 sec before resend to generative ai",ex,False)
            time.sleep(15)  # <- Generative AI rate limit: up to 3 requests per minute
           
            if attempts > 0:
                return self.ask(prompt, attempts - 1)
            return 
