from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import classification_report
import numpy as np
import torch

from utils import load_config, load_dataset

#### Load model
model_path = '5CD-AI/Vietnamese-Sentiment-visobert'
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")


if __name__ == "__main__":
    config = load_config("../config.yaml")
    dataset = load_dataset(config.datasets.social_comments.path)
    
    # Convert dataset column to list of strings
    sentences = list(dataset['train']['sentence'])
    labels = list(dataset['train']['sentiment'])
    
    # Tokenize the sentences
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    # Shift labels from [-1, 0, 1] to [0, 1, 2]
    labels = [label + 1 for label in labels]
    
    batch_size = 64
    predictions = []
    
    for i in range(0, len(labels), batch_size):
        batch_input_ids = input_ids[i:i+batch_size].to("cuda")
        batch_attention_mask = attention_mask[i:i+batch_size].to("cuda")
        batch_labels = labels[i:i+batch_size]
        
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    print(classification_report(labels, predictions))