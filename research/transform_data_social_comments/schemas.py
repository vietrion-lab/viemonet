from pydantic import BaseModel


class FModelSchema(BaseModel):
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    system_prompt: str

class TransformConfig(BaseModel):
    foundation_model: FModelSchema
    dataset: str