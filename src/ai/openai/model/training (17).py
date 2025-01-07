## \file /src/ai/openai/model/training (17).py
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


""" OpenAI Model Class for handling communication with the OpenAI API and training the model.  """



from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import header
from src.logger.logger import logger
from src import gs
from src.utils.jjson import j_loads_ns, j_dumps
from openai import OpenAI

class Model:
    """OpenAI Model Class"""
    model: str = "gpt-3.5-turbo-instruct"
    client: OpenAI
    current_job_id: str
    assistant = None
    thread = None

    def __init__(self):
        """Initialize the Model object with API key and assistant ID from settings."""
        self.client = OpenAI(api_key=gs.credentials.openai.project_api)
        self.current_job_id = None
        self.assistant = self.client.beta.assistants.retrieve(gs.credentials.openai.assistant_id)
        self.thread = self.client.beta.threads.create()



    def stream_w(self, data: List[Dict[str, Union[str, None]]] = None, data_file_path: str | Path = None):
        """ Send prepared chat data 
        see `src.suppliers.chat_gpt.chat_gpt.py`
        """
        ...
        def get_incremented_filename(file_path: Path) -> Path:
            """Generate a new filename with an incremented number if the file already exists, only once per input file."""
            base_name = file_path.stem
            extension = file_path.suffix
            i = 1
            new_file_path = file_path

            # Increment the filename only if it exists
            if new_file_path.exists():
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
            # Determine if we need to increment the filename
            new_file_path = get_incremented_filename(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return



            try:
                # Initialize or load the CSV file
                df_existing = pd.read_csv(new_file_path) if file_path.exists() else pd.DataFrame(columns=["role", "content", "sentiment"])
            except Exception as ex:
                logger.error(f"Error loading CSV file: {file_path}", ex)
                return

            # Initialize variables
            last_user_message = None
            last_assistant_message = None

            # Process the CSV file in chunks
            for chunk in pd.read_csv(file_path, chunksize=100):
                for _, row in chunk.iterrows():
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
                            # Save the last user message for later use
                            last_user_message = message_dict

                        elif role == 'assistant':
                            # Save the last assistant message for later use
                            last_assistant_message = message_dict

                            if last_user_message:
                                # Prepare the interaction data with relevant messages only
                                response = self.send_message(last_user_message["content"])

                                # Prepare the interaction data
                                interaction = [
                                    {"role": "user", "content": last_user_message["content"], "sentiment": last_user_message["sentiment"]},
                                    {"role": "assistant", "content": last_assistant_message["content"], "sentiment": last_assistant_message["sentiment"]},
                                    {"role": "assistant", "content": response, "sentiment": "negative"}
                                ]

                                # Convert interaction to DataFrame and append to the existing DataFrame
                                df_interaction = pd.DataFrame(interaction)
                                df_existing = pd.concat([df_existing, df_interaction], ignore_index=True)

                                # Save the updated DataFrame to the incremented filename
                                try:
                                    logger.info(f"Updating {new_file_path}")
                                    df_existing.to_csv(new_file_path, index=False)
                                except Exception as ex:
                                    logger.error(f"Error saving CSV file: {new_file_path}", ex)

                                # Reset the user message and assistant message
                                last_user_message = None
                                last_assistant_message = None

            # Handle any remaining messages
            if last_user_message and last_assistant_message:
                response = self.send_message(last_user_message["content"])

                # Prepare the interaction data
                interaction = [
                    {"role": "user", "content": last_user_message["content"], "sentiment": last_user_message["sentiment"]},
                    {"role": "assistant", "content": last_assistant_message["content"], "sentiment": last_assistant_message["sentiment"]},
                    {"role": "assistant", "content": response, "sentiment": "negative"}
                ]

                # Convert interaction to DataFrame and append to the existing DataFrame
                df_interaction = pd.DataFrame(interaction)
                df_existing = pd.concat([df_existing, df_interaction], ignore_index=True)

                # Save the updated DataFrame to the incremented filename
                try:
                    logger.info(f"Updating {new_file_path}")
                    df_existing.to_csv(new_file_path, index=False)
                except Exception as ex:
                    logger.error(f"Error saving CSV file: {new_file_path}", ex)

        else:
            logger.error("No data file path provided")








    def _send_batch(self, messages: List[Dict[str, Union[str, None]]]):
        """ Send a batch of messages to the OpenAI API"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
        except Exception as ex:
            logger.error("An error occurred while sending the batch:", ex)

    def send_message(self, message: str) -> str:
        """ Send a message to the model and return the response"""
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=message,
                max_tokens=4000,
                temperature=0
            )
            return response.choices[0].text.strip()
        except Exception as ex:
            logger.error("An error occurred while sending the message:", ex)
            return ""

    def train(self, data: str = None, data_dir: Path | str = None, data_file: Path | str = None, positive: bool = True) -> str | None:
        """Train the model on the specified data or directory."""
        try:
            documents = []
            if data:
                documents = j_loads(data)
            elif data_file:
                documents = j_loads(data_file)
            elif data_dir:
                for file_path in Path(data_dir).glob('*.json'):
                    documents.extend(j_loads(file_path))
            else:
                raise ValueError("No data source provided")

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
    model.stream_w(data_file_path=Path(gs.path.google_drive / 'chat_gpt' / 'conversation' / 'all_conversations.csv'))

if __name__ == "__main__":
    main()
