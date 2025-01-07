## \file /src/ai/openai/model/training (11).py
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
import pandas as pd
import header
from src import gs
from src.utils.jjson import j_loads_ns, j_dumps
from src.logger.logger import logger
from openai import OpenAI




from typing import List, Dict, Union
import pandas as pd
from pathlib import Path
from src.logger.logger import logger

class Model:
    """OpenAI Model Class"""
    ...
    model:str = "gpt-3.5-turbo-instruct"
    client:OpenAI
    current_job_id:str
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
        if data is None and data_file_path is None:
            logger.error("No data provided")
            return

        if data_file_path:
            file_path = Path(data_file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return

            # Load the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path)
            except Exception as ex:
                logger.error(f"Error loading CSV file: {file_path}", ex)
                return

            # Process the DataFrame in chunks
            messages = []
            current_message_length = 0
            last_role = None

            for _, row in df.iterrows():
                role = row.get('role')
                content = row.get('content')
                sentiment = row.get('sentiment', 'neutral')

                if role and content:
                    message_dict = {
                        "role": role,
                        "content": content,
                        "sentiment": sentiment
                    }

                    if last_role and last_role != role:
                        # Add the message pair to the batch
                        messages.append(message_dict)
                        message_length = len(str(message_dict))
                        
                        # Check if adding this message exceeds the limit
                        if current_message_length + message_length > 2000:
                            # Send the batch
                            self._send_batch(messages)

                            # Reset batch
                            messages = []
                            current_message_length = 0

                        current_message_length += message_length
                    else:
                        # Start a new message pair
                        messages = [message_dict]  # Initialize with the current message
                        response = self.send_message(messages)
                        _iteraction:list = [
                        {"role": "user","content": user_content,"sentiment": sentiment  },
                        {"role":"assistant","content": response, "sentiment":"negative"}
                            ]
                        j_dumps(_iteraction,Path(gs.path.google_drive / 'chat_gpt' / 'conversation' / '1st_iteracted.json'),mode='a')
                        
                        current_message_length = len(str(message_dict))

                    last_role = role

            # Send remaining messages in batch
            if messages:
                self._send_batch(messages)

        elif data:
            # Handle case where data is directly provided
            messages = []
            current_message_length = 0

            for item in data:
                messages.extend(item)
                message_length = len(str(item))
                
                # Check if adding this message exceeds the limit
                if current_message_length + message_length > 2000:
                    # Send the batch
                    self._send_batch(messages)

                    # Reset batch
                    messages = []
                    current_message_length = 0

                # Add message to batch
                messages.append(item)
                current_message_length += message_length

            # Send remaining messages in batch
            if messages:
                self._send_batch(messages)

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


    # Other methods (send_message, train, etc.) as previously defined

    def send_message(self,message:str):
        """Отрпавка сообщения в модель """
        message = self.client.completions.create(
          model=self.model,
          prompt=message,
          max_tokens=4000,
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
