from pydantic import BaseModel


class SocialCommentDetail(BaseModel):
    path: str
    text_field: str
    label_field: str
    name: str
    description: str
    
class DatasetConfig(BaseModel):
    social_comments: SocialCommentDetail
    vietnamese_comments: SocialCommentDetail
    merged_dataset: SocialCommentDetail
    augmentation_dataset: SocialCommentDetail

class ConfigSchema(BaseModel):
    datasets: DatasetConfig