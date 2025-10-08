import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Custom Dataset for emotion-based sentiment analysis.
    Wraps the preprocessed data (tensors and lists) for use with PyTorch DataLoader.
    """
    
    def __init__(self, data_dict):
        """
        Args:
            data_dict: Dictionary with keys 'input_ids', 'attention_mask', 'emotions', 'labels'
        """
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']
        self.emotions = data_dict['emotions'] if 'emotions' in data_dict else None
        self.labels = data_dict['labels']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
        
        if self.emotions is not None:
            data['emotions'] = self.emotions[idx]
        
        return data
