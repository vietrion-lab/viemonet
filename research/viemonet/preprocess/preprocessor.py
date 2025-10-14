from typing import List
import torch

from viemonet.preprocess.preprocess_pipeline import PreprocessPipeline
from viemonet.constant import METHOD


class TrainingPreprocessor:
    def __init__(self, method, model_name, foundation_model_name=None):
        self.method = method
        self.model_name = model_name
        self.pipeline = PreprocessPipeline(method=method, foundation_model_name=foundation_model_name)

    def normalize_labels(self, labels: list) -> list:
        """Normalize labels to numeric format [-1, 0, 1]. Handles strings and numbers."""
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        normalized = []
        
        for label in labels:
            if isinstance(label, str):
                normalized.append(label_map.get(label.lower().strip(), None))
                if normalized[-1] is None:
                    raise ValueError(f"Unknown label: '{label}'")
            else:
                normalized.append(int(label))
        
        return normalized

    def shift_labels(self, labels: List[int]) -> List[int]:
        """Shift labels from [-1, 0, 1] to [0, 1, 2] for softmax."""
        return [label + 1 for label in labels]

    def __call__(self, raw_data):
        """Process raw data and prepare it for training."""
        processed_data = {}
        
        for split in ['train', 'validation', 'test']:
            texts = raw_data[split]['text']
            labels = raw_data[split]['label']

            # Filter out None labels and keep texts aligned
            valid_data = [(t, l) for t, l in zip(texts, labels) if l is not None]
            if len(valid_data) < len(texts):
                print(f"Warning: Filtered {len(texts) - len(valid_data)} None labels from {split}")
            
            texts, labels = zip(*valid_data) if valid_data else ([], [])
            
            # Normalize and shift labels: strings/numbers -> [-1,0,1] -> [0,1,2]
            labels = self.shift_labels(self.normalize_labels(list(labels)))
            
            # Process texts through pipeline
            pipeline_output = self.pipeline(list(texts))
            
            # Store processed data
            processed_data[split] = {
                'input_ids': pipeline_output['input_ids'],
                'attention_mask': pipeline_output['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long)
            }
            
            if self.method == METHOD[0]:
                processed_data[split]['emotions'] = pipeline_output['emotions']
        
        return processed_data