import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from viemonet.constant import METHOD


@dataclass
class EmotionDataCollator:
    """
    Custom data collator for emotion-based sentiment analysis.
    Handles batching of mixed data types (tensors and lists of strings).
    """
    
    method: str = METHOD[0]  # Default to 'seperate_emotion'
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            features: List of dictionaries with keys 'input_ids', 'attention_mask', 'emotions', 'labels'
            
        Returns:
            Batched dictionary with properly stacked tensors and lists
        """
        # Filter out None features
        features = [f for f in features if f is not None]
        
        if len(features) == 0:
            raise ValueError("All features in the batch are None. Check dataset __getitem__ method.")
        
        batch = {}
        
        # Don't filter - all features MUST have these keys
        # If any feature is missing a key, we want it to fail with a clear error
        batch['ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attn'] = torch.stack([f['attention_mask'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        
        # Only include emotions if method is 'seperate_emotion' (METHOD[0])
        if self.method == METHOD[0] and len(features) > 0 and 'emotions' in features[0]:
            batch['emo'] = [f.get('emotions', None) for f in features]

        return batch
