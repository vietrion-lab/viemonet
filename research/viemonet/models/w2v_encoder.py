"""Word2Vec encoder that trains on dataset and provides embedding features."""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
from typing import List, Optional
from gensim.models import Word2Vec
from underthesea import word_tokenize


class W2VEncoder(nn.Module):
    """Word2Vec-based encoder that trains on dataset text and provides features.
    
    This encoder:
    1. Trains a Word2Vec model on the provided dataset (or loads existing)
    2. Converts tokenized input_ids back to text
    3. Word-tokenizes with underthesea
    4. Averages W2V embeddings for each word
    5. Returns a HuggingFace-like output with .last_hidden_state
    """
    
    def __init__(self, corpus_texts: Optional[List[str]] = None, 
                 cache_path: Optional[str] = None,
                 vector_size: int = 768,
                 window: int = 5,
                 min_count: int = 2,
                 epochs: int = 10,
                 workers: int = 4,
                 sg: int = 0,
                 tokenizer=None):
        """Initialize W2V encoder.
        
        Args:
            corpus_texts: List of raw text strings to train W2V on (optional if loading from cache)
            cache_path: Path to save/load trained W2V model
            vector_size: Dimension of word vectors (default 768 to match BERT)
            window: W2V context window size
            min_count: Minimum word frequency
            epochs: Training epochs for W2V
            workers: Number of parallel workers
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
            tokenizer: HuggingFace tokenizer for decoding input_ids
        """
        super().__init__()
        self.vector_size = vector_size
        self.tokenizer = tokenizer
        self.cache_path = Path(cache_path) if cache_path else None
        self.w2v_model = None
        
        # Try to load cached model first
        if self.cache_path and self.cache_path.exists():
            print(f"📥 Loading cached W2V model from {self.cache_path}")
            self.w2v_model = Word2Vec.load(str(self.cache_path))
            print(f"✅ Loaded W2V: vocab_size={len(self.w2v_model.wv)}, vector_size={self.w2v_model.vector_size}")
        elif corpus_texts:
            # Train new W2V model
            print(f"🔨 Training W2V model on {len(corpus_texts)} texts...")
            self._train_w2v(corpus_texts, vector_size, window, min_count, epochs, workers, sg)
        else:
            raise ValueError("Must provide either corpus_texts (for training) or valid cache_path (for loading)")
    
    def _train_w2v(self, corpus_texts: List[str], vector_size: int, 
                   window: int, min_count: int, epochs: int, workers: int, sg: int):
        """Train Word2Vec on corpus texts."""
        # Tokenize corpus with underthesea
        print(f"🔤 Tokenizing {len(corpus_texts)} texts with underthesea...")
        tokenized_corpus = []
        for i, text in enumerate(corpus_texts):
            if i % 1000 == 0 and i > 0:
                print(f"   Tokenized {i}/{len(corpus_texts)} texts...")
            try:
                tokens = word_tokenize(text.lower())
                if tokens:  # Skip empty
                    tokenized_corpus.append(tokens)
            except Exception:
                continue
        
        print(f"✅ Tokenized {len(tokenized_corpus)} documents")
        print(f"🔧 Training W2V (vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}, sg={sg})...")
        
        # Train Word2Vec
        self.w2v_model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=sg,
        )
        
        print(f"✅ W2V trained: vocab_size={len(self.w2v_model.wv)}, vector_size={self.w2v_model.vector_size}")
        
        # Save to cache if path provided
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.w2v_model.save(str(self.cache_path))
            print(f"💾 Saved W2V model to {self.cache_path}")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass: convert input_ids to W2V features.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) - tokenized by HF tokenizer
            attention_mask: Tensor of shape (batch_size, seq_len) - not used but kept for compatibility
            
        Returns:
            Object with .last_hidden_state attribute of shape (batch_size, seq_len, vector_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, seq_len, self.vector_size, device=device)
        
        # Process each sample in batch
        for i in range(batch_size):
            # Decode input_ids back to text
            if self.tokenizer:
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            else:
                # Fallback: treat as space-separated tokens
                text = " ".join(map(str, input_ids[i].tolist()))
            
            # Word tokenize with underthesea
            try:
                tokens = word_tokenize(text.lower())
            except Exception:
                tokens = text.lower().split()
            
            # Get W2V embedding for each token
            for j, token in enumerate(tokens[:seq_len]):  # Truncate to seq_len
                if token in self.w2v_model.wv:
                    vec = torch.tensor(self.w2v_model.wv[token], dtype=torch.float32, device=device)
                    embeddings[i, j] = vec
                # else: leave as zeros (OOV words)
        
        # Return HuggingFace-like output
        class W2VOutput:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states
        
        return W2VOutput(embeddings)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict (W2V model is saved separately via gensim)."""
        return {
            'vector_size': self.vector_size,
            'cache_path': str(self.cache_path) if self.cache_path else None,
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict (W2V model loaded from cache_path)."""
        self.vector_size = state_dict.get('vector_size', 768)
        cache_path = state_dict.get('cache_path')
        if cache_path and Path(cache_path).exists():
            self.cache_path = Path(cache_path)
            self.w2v_model = Word2Vec.load(str(self.cache_path))
