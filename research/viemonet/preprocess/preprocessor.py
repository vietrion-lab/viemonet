from typing import List
import torch

from viemonet.preprocess.preprocess_pipeline import PreprocessPipeline
from viemonet.constant import METHOD


class TrainingPreprocessor:
    def __init__(self, method, foundation_model_name=None):
        self.method = method
        self.pipeline = PreprocessPipeline(method=method, foundation_model_name=foundation_model_name)

    def shift_labels(self, labels: List[int]) -> List[int]:
        """
        Shift labels from [-1, 0, 1] to [0, 1, 2] for softmax.
        
        Mapping:
        -1 (negative) -> 0
         0 (neutral)  -> 1
         1 (positive) -> 2
        
        Args:
            labels: List of labels in [-1, 0, 1] range
            
        Returns:
            List of labels in [0, 1, 2] range
        """
        return [label + 1 for label in labels]

    def __call__(self, raw_data):
        """
        Process raw data and prepare it for training.
        
        Args:
            raw_data: Dataset with 'train', 'validation', and 'test' splits
            
        Returns:
            Processed data as torch tensors for each split
        """
        processed_data = {}
        
        for split in ['train', 'validation', 'test']:
            # Get text and labels from the dataset
            text_field = 'sentence'
            label_field = 'sentiment'
            
            texts = raw_data[split][text_field]
            labels = raw_data[split][label_field]

            # Shift labels from [-1, 0, 1] to [0, 1, 2]
            shifted_labels = self.shift_labels(labels)
            
            # Process texts to separate comments and emotions
            # Pipeline now returns a dict with 'input_ids', 'attention_mask', 'emotions'
            pipeline_output = self.pipeline(texts)
            
            # Convert labels to tensor
            labels_tensor = torch.tensor(shifted_labels, dtype=torch.long)
            
            # Store as tensors (input_ids and attention_mask are already tensors from pipeline)
            processed_data[split] = {
                'input_ids': pipeline_output['input_ids'],
                'attention_mask': pipeline_output['attention_mask'],
                'labels': labels_tensor
            }
            
            if self.method == METHOD[0]:
                processed_data[split]['emotions'] = pipeline_output['emotions']
        
        return processed_data