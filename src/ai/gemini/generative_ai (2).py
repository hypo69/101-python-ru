## \file /src/ai/gemini/generative_ai (2).py
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
from typing import Optional
import os
import pathlib
import textwrap
import google.generativeai as genai

from src.logger.logger import logger
from src import gs
from src.utils.printer import pprint

class GoolgeGenerativeAI:
    """"""
    model:genai.GenerativeModel
    system_instruction:str
    
    def __init__(self, system_instruction:Optional[str]=None):
        """ """
        #genai.configure(api_key = os.environ["API_KEY"])
        genai.configure(api_key = gs.credentials.googleai.api_key)

        # Using `response_mime_type` requires either a Gemini 1.5 Pro or 1.5 Flash model
        self.system_instruction = system_instruction
        self.model = genai.GenerativeModel('gemini-1.5-flash',
                                      # Set the `response_mime_type` to output JSON
                                      generation_config={"response_mime_type": "application/json"},
                                      system_instruction = system_instruction if system_instruction else None)


    def ask(self, prompt:str, attempts: int = 5):
        """"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as ex:
            logger.error(f"Generative AI prompt{pprint(prompt)}\n{attempts=}", ex, True)
            time.sleep(15)  # <- принимает до 5 запросов в минуту
            if attempts >0:
                self.ask(prompt, attempts - 1)