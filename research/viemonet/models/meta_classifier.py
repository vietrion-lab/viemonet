import torch
import torch.nn as nn


class MetaClassifier(nn.Module):
    """
    Meta classifier that combines comment sentiment, emotion sentiment, and rule features
    to make final sentiment predictions.
    """
    
    def __init__(self, input_dim=15, hidden_dim=32, output_dim=3):
        """
        Args:
            input_dim: Input dimension (3 comment + 3 emotion + 9 rule features = 15)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (3 classes: positive, neutral, negative)
        """
        super(MetaClassifier, self).__init__()
        
        # Deeper network with residual-like connections
        self.meta_cls_fc1 = nn.Linear(input_dim, hidden_dim * 2)  # 15 -> 64
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        
        self.meta_cls_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)  # 64 -> 32
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.meta_cls_fc3 = nn.Linear(hidden_dim, output_dim)  # 32 -> 3
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, comment_probs, emotion_probs, rule_feature, labels=None):
        """
        Forward pass combining all features.
        
        Args:
            comment_probs: Comment sentiment probabilities (batch_size, 3)
            emotion_probs: Emotion sentiment probabilities (batch_size, 3)
            rule_feature: Rule-based feature matrix (batch_size, 3, 3)
            labels: Ground truth labels (batch_size,) - optional
            
        Returns:
            dict with 'loss', 'logits', and 'probs'
        """
        batch_size = comment_probs.shape[0]
        
        # Flatten rule_feature from (batch_size, 3, 3) to (batch_size, 9)
        rule_feature_flat = rule_feature.view(batch_size, -1)
        
        # Concatenate all features: (batch_size, 3+3+9=15)
        combined_features = torch.cat([comment_probs, emotion_probs, rule_feature_flat], dim=-1)
        
        # Pass through deeper network
        x = self.meta_cls_fc1(combined_features)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.meta_cls_fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        logits = self.meta_cls_fc3(x)

        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        
        return {"loss": loss, "logits": logits, "probs": probs}