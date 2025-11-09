from pydantic import BaseModel


class SocialCommentDetail(BaseModel):
    path: str
    text_field: str
    label_field: str
    name: str
    description: str
    
class DatasetConfig(BaseModel):
    social_comments: SocialCommentDetail
    augmentation_dataset: SocialCommentDetail
    vlsp2016_dataset: SocialCommentDetail

class ConfigSchema(BaseModel):
    datasets: DatasetConfig