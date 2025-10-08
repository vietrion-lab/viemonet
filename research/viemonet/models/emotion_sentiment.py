from datasets import load_dataset
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Tuple

from viemonet.config import config


class EmotionSentiment:
    """
    Class to load and manage emotion sentiment data for emoticons and emojis.
    Converts sentiment scores to probability distributions over positive, neutral, and negative classes.
    Merges emoticons and emojis into a single unified DataFrame.
    """
    
    def __init__(self):
        self.emotion_df = None  # Unified DataFrame for both emoticons and emojis
        self.softmax = nn.Softmax(dim=-1)
        self._load_data()
    
    def _load_data(self):
        """Load and process emoticon and emoji sentiment data, then merge them."""
        # Load datasets
        emoticon_sentiment = load_dataset(
            'csv',
            data_files=config.emotion_knowledge_base.emoticon_sentiment
        )['train'].to_pandas()

        emoji_sentiment = load_dataset(
            'csv',
            data_files=config.emotion_knowledge_base.emoji_sentiment
        )['train'].to_pandas()

        # Process emoticon data
        emoticon_df = emoticon_sentiment[['emoticon', 'sentiment_score']]
        emoticon_df['sentiment_score'] = emoticon_df['sentiment_score'].astype(float)
        
        emoticon_df['positive'] = 0.0
        emoticon_df['neutral'] = 0.0
        emoticon_df['negative'] = 0.0
        
        self._calculate_emoticon_probabilities(emoticon_df)
        
        # Drop sentiment_score column and rename emoticon to symbol
        emoticon_df = emoticon_df.drop(columns=['sentiment_score'])
        emoticon_df = emoticon_df.rename(columns={'emoticon': 'symbol'})
        emoticon_df['type'] = 'emoticon'
        
        # Process emoji data
        emoji_df = emoji_sentiment[['Unicode codepoint', 'Positive', 'Neutral', 'Negative']]
        emoji_df = emoji_df.rename(columns={
            'Unicode codepoint': 'symbol', 
            'Positive': 'positive', 
            'Neutral': 'neutral', 
            'Negative': 'negative'
        })
        emoji_df = emoji_df.astype({"positive": float, "neutral": float, "negative": float})
        emoji_df['type'] = 'emoji'
        
        self._calculate_emoji_probabilities(emoji_df)
        
        # Merge both DataFrames
        self.emotion_df = pd.concat([emoticon_df, emoji_df], ignore_index=True)
        
        # Reorder columns: symbol, type, positive, neutral, negative
        self.emotion_df = self.emotion_df[['symbol', 'type', 'positive', 'neutral', 'negative']]
    
    def _calculate_emoticon_probabilities(self, emoticon_df):
        """Calculate probabilities for emoticons from sentiment scores."""
        for idx, row in emoticon_df.iterrows():
            score = row['sentiment_score']
            
            if score > 0:
                pos_strength = abs(score) ** 1.5
                scores = torch.tensor([pos_strength, 0.3 * (1 - abs(score)), 0.0], dtype=torch.float32)
            elif score < 0:
                neg_strength = abs(score) ** 1.5
                scores = torch.tensor([0.0, 0.3 * (1 - abs(score)), neg_strength], dtype=torch.float32)
            else:
                scores = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            
            probs = self.softmax(scores).numpy()
            emoticon_df.loc[idx, ['positive', 'neutral', 'negative']] = probs
    
    def _calculate_emoji_probabilities(self, emoji_df):
        """Calculate probabilities for emojis using softmax on existing scores."""
        for idx, row in emoji_df.iterrows():
            scores = torch.tensor([row['positive'], row['neutral'], row['negative']], dtype=torch.float32)
            probs = self.softmax(scores).numpy()
            emoji_df.loc[idx, ['positive', 'neutral', 'negative']] = probs
    
    def get_emotion_data(self) -> pd.DataFrame:
        """Get the unified emotion DataFrame with probability distributions."""
        return self.emotion_df
    
    def get_emoticon_data(self) -> pd.DataFrame:
        """Get only the emoticon data (filtered from unified DataFrame)."""
        return self.emotion_df[self.emotion_df['type'] == 'emoticon'].copy()
    
    def get_emoji_data(self) -> pd.DataFrame:
        """Get only the emoji data (filtered from unified DataFrame)."""
        return self.emotion_df[self.emotion_df['type'] == 'emoji'].copy()
    
    def get_all_emoticons(self) -> List[str]:
        """Get a list of all text emoticons."""
        return self.emotion_df[self.emotion_df['type'] == 'emoticon']['symbol'].tolist()
    
    def get_all_emojis(self) -> List[str]:
        """Get a list of all emojis."""
        return self.emotion_df[self.emotion_df['type'] == 'emoji']['symbol'].tolist()
    
    def get_all_symbols(self) -> List[str]:
        """Get a list of all emotion symbols (both emoticons and emojis)."""
        return self.emotion_df['symbol'].tolist()
    
    def get_data(self) -> pd.DataFrame:
        """Get the unified emotion DataFrame."""
        return self.emotion_df


# if __name__ == '__main__':
#     # Set pandas display options to show all rows
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)
    
#     emotion_sentiment = EmotionSentiment()
    
#     print("All Emoticons:")
#     print(emotion_sentiment.get_all_emoticons())
#     print("\nAll Emojis:")
#     print(emotion_sentiment.get_all_emojis())
#     print("\n" + "="*80 + "\n")
    
#     emotion_df = emotion_sentiment.get_data()
#     print("Unified Emotion DataFrame:")
#     print(emotion_df)
#     print(f"\nTotal emotions: {len(emotion_df)}")
#     print(f"Emoticons: {len(emotion_df[emotion_df['type'] == 'emoticon'])}")
#     print(f"Emojis: {len(emotion_df[emotion_df['type'] == 'emoji'])}")