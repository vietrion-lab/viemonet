from pydantic import BaseModel
from typing import List


class EmoticonSchema(BaseModel):
    emoticon: List[str]
    sentiment_score: List[float]

class EmojiSchema(BaseModel):
    unicode: List[str]
    positive: List[float]
    neutral: List[float]
    negative: List[float]

class EmotionSchema(BaseModel):
    symbol: str
    positive_score: float
    neutral_score: float
    negative_score: float