from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

class OpenAICore:
    def __init__(
        self,
        model: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        system_prompt: str
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt
        
    def __call__(self, comment: str):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "integer",
                                "enum": [-1, 0, 1],
                                "description": "-1 = Negative, 0 = Neutral, 1 = Positive"
                            }
                        },
                        "required": ["sentiment"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Comment: '{comment}'"
                }
            ]
        )

        result = json.loads(response.choices[0].message.content)
        return result["sentiment"]
