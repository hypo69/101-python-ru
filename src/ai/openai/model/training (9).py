## \file /src/ai/openai/model/training (9).py
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


""" HERE SHOULD BE A DESCRIPTION OF THE MODULE OPERATION ! """



from pathlib import Path
from typing import List, Dict
import header
from src import gs
from src.utils.jjson import j_loads_ns
from src.logger.logger import logger
from openai import OpenAI

class Model:
    """OpenAI Model Class"""

    def __init__(self):
        """Initialize the Model object with API key and assistant ID from settings."""
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant = self.client.beta.assistants.retrieve(gs.credentials.openai.assistant_id)
        self.thread = self.client.beta.threads.create()

    def stream_w(self, data: List[Dict] = None, data_file_path: str | Path = None):
        """ Send prepared chat data 
        see `src.suppliers.chat_gpt.chat_gpt.py`
        """
        if data is None and data_file_path is None:
            logger.error("No data provided")
            return

        if data_file_path:
            file_path = Path(data_file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return

            # Initialize variables
            messages = []
            current_message_length = 0
            message_batch = []

            # Read the file line-by-line
            with file_path.open('r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        # Parse JSON line
                        try:
                            line_data = j_loads(line)
                            role = line_data.get('role')
                            content = line_data.get('content')
                            sentiment = line_data.get('sentiment')

                            if role and content:
                                message = f"{role}: {content}"
                                message_length = len(message)

                                # Check if adding this message exceeds the limit
                                if current_message_length + message_length > 2000:
                                    # Send the batch
                                    self._send_batch(message_batch)

                                    # Reset batch
                                    message_batch = []
                                    current_message_length = 0

                                # Add message to batch
                                message_batch.append(message)
                                current_message_length += message_length

                        except Exception as ex:
                            ...
                            logger.error(f"Error processing line: {line.strip()}", ex)

            # Send remaining messages in batch
            if message_batch:
                self._send_batch(message_batch)

        elif data:
            # Handle case where data is directly provided
            counter = 0
            messages = []
            for item in data:
                messages.extend(item)
                counter += 1
                if counter > 3:
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            print(chunk.choices[0].delta.content, end="")
                    counter = 0
                    messages = []

    def _send_batch(self, messages: List[str]):
        """ Send a batch of messages to the API """
        try:
            if messages:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": msg} for msg in messages],
                    temperature=0,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        print(chunk.choices[0].delta.content, end="")
        except Exception as ex:
            logger.error("An error occurred while sending the batch:", ex)

    
    
    def send_message(self,message:str):
        """Отрпавка сообщения в модель """
        message = self.client.completions.create(
          model="gpt-3.5-turbo-instruct",
          prompt=message,
          max_tokens=2400,
          temperature=0
        )
        return message.choices[0].text.strip()


    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
        try:
            # if data:
            #     documents = j_loads(data)
            # elif data_file:
            #     documents = j_loads(data_file)
            # elif data_dir:
            #     documents = []
            #     for file_path in Path(data_dir).glob('*.json'):
            #         documents.extend(j_loads(file_path))
            # else:
            #     raise ValueError("No data source provided")

            response = self.client.Training.create(
                model="text-davinci-003",
                documents=data,
                labels=["positive" if positive else "negative"] * len(documents),
                show_progress=True
            )
            self.current_job_id = response.id
            return response.id

        except Exception as ex:
            logger.error("An error occurred during training:", ex)
            return
        
    def save_job_id(self, job_id: str, description: str, filename: str = "job_ids.json"):
        """Save the job ID with description to a file."""
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
        """Update the assistant to include the file_search tool and link the vector store."""
        try:
            updated_assistant = self.client.Assistant.update(
                assistant_id,
                tool="file_search",
                file_search_vector_store=vector_store_id
            )
            logger.info(f"Assistant updated successfully: {updated_assistant}")
        except Exception as ex:
            logger.error("An error occurred while updating the assistant:", ex)

def main():
    """Main function to execute the training and assistant management."""
    model = Model()
    model.stream_w(data_file_path = Path(gs.path.google_drive / 'chat_gpt' / 'conversation' / 'all_conversations.csv'))

if __name__ == "__main__":
    main()
