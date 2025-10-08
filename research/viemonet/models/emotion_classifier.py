import emoji
import torch
import torch.nn as nn

from viemonet.models.emotion_sentiment import EmotionSentiment


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.emo_sent = EmotionSentiment().get_data()

    def forward(self, emos, device=None):
        """
        Get sentiment scores for given emotions.
        
        Args:
            emos: List of lists of emotion symbols (e.g., [[':-)', '😊'], [':('], ...])
            device: Device to create tensors on (if None, uses CPU)
            
        Returns:
            dict with 'probs': torch.Tensor of shape (batch_size, 3) with [positive, neutral, negative] scores
        """
        batch_size = len(emos)
        
        # Determine device - use provided device or CPU as fallback
        target_device = device if device is not None else torch.device('cpu')
        
        scores = torch.zeros(batch_size, 3, dtype=torch.float32, device=target_device)
        
        for i, emo_list in enumerate(emos):
            if not emo_list:
                scores[i] = torch.tensor([0.0, 0.0, 0.0], device=target_device)
            else:
                # Accumulate scores for all emotions in the list
                pos_sum = 0.0
                neu_sum = 0.0
                neg_sum = 0.0
                count = 0
                
                for emo in emo_list:
                    # Look up the emotion in the dataframe
                    matched = self.emo_sent[self.emo_sent['symbol'] == emo]
                    
                    if not matched.empty:
                        pos_sum += matched.iloc[0]['positive']
                        neu_sum += matched.iloc[0]['neutral']
                        neg_sum += matched.iloc[0]['negative']
                        count += 1
                
                # Set scores after processing all emotions
                if count > 0:
                    scores[i] = torch.tensor([pos_sum / count, neu_sum / count, neg_sum / count], device=target_device)
                else:
                    scores[i] = torch.tensor([0.0, 0.0, 0.0], device=target_device)
        
        return {"probs": scores}