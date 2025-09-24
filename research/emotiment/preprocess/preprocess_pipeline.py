from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from pathlib import Path
import torch
import torch.nn as nn
import emoji
import re
from typing import Dict

from emotiment.config import config
from emotiment.dataset import emoji_dataset
from emotiment.constant.training_constant import EMOJI2VEC_METHOD, EMOJI2DESCRIPTION_METHOD


class PreprocessPipeline:
    def __init__(self, tokenizer=None, method=None):
        """Preprocessing + (optional) emoji2vec token augmentation.

        When method == EMOJI2VEC_METHOD:
          1. Load raw emoji2vec (emoji + 300-dim vector) from res/emoji2vec.txt.
          2. Add every emoji token (that is not already present) into the tokenizer vocab.
          3. Create a lightweight Linear projection 300 -> hidden_size (defaults to HF config.hidden_size or 768).
          4. Pre-compute projected vectors and keep them in `projected_emoji_vectors` (dict: emoji -> torch.Tensor).

        NOTE: This class does NOT itself resize or modify the foundation model's embedding
        layer. After constructing this pipeline you should, when building the model, call
        `model.resize_token_embeddings(len(pipeline.tokenizer))` and (optionally) copy the
        projected vectors into the corresponding rows of the model embedding weight.
        Keeping that separation makes this pipeline simple and side‑effect free.
        """
        assert method in {EMOJI2VEC_METHOD, EMOJI2DESCRIPTION_METHOD}, f"Unknown method: {method}"
        self.method = method

        # ------------- Load description dataset (used for EMOJI2DESCRIPTION_METHOD) -------------
        self.emojis = emoji_dataset(config)
        try:
            # Updated for Vietnamese emoticon dictionary format
            self.emoji_map: Dict[str, str] = {
                row["emoticon_code"]: row["description"].strip().lower() for row in self.emojis
                if row.get("emoticon_code") and row.get("description")
            }
        except Exception:
            self.emoji_map = {}

        # ------------- Tokenizer -------------
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.foundation_model.tokenizer_name
        )

        # ------------- Emoji2Vec handling -------------
        self.emoji2vec_raw = {}
        self.projected_emoji_vectors = {}
        if self.method == EMOJI2VEC_METHOD:
            self._init_emoji2vec()
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")

    # ---------------- Internal helpers ----------------
    def _init_emoji2vec(self):
        """Load emoji2vec.txt, add tokens, and build projected vectors.

        File format (first line: `<count> <dim>` then: `<emoji> v1 v2 ... v300`).
        Keeps parsing extremely small & robust; silently skips malformed lines.
        """
        root = Path(__file__).resolve().parents[3]  # repo root
        emoji2vec_path = root / "res" / "emoji2vec.txt"
        if not emoji2vec_path.exists():
            return  # graceful: nothing to do

        try:
            with emoji2vec_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split()
                # header like: 1661 300 (ignore if malformed)
                if len(header) >= 2 and header[1].isdigit():
                    expected_dim = int(header[1])
                else:
                    expected_dim = 300
                for line in f:
                    parts = line.rstrip().split()
                    if len(parts) < expected_dim + 1:
                        continue
                    emo = parts[0]
                    try:
                        vec = [float(x) for x in parts[1:expected_dim+1]]
                    except ValueError:
                        continue
                    if len(vec) != expected_dim:
                        continue
                    self.emoji2vec_raw[emo] = vec
        except Exception:
            # If anything goes wrong, leave structures empty
            self.emoji2vec_raw = {}
            return

        # Add new emoji tokens to tokenizer vocab (avoid duplicates)
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        new_emojis = [e for e in self.emoji2vec_raw.keys() if e not in existing_vocab]
        if new_emojis:
            self.tokenizer.add_tokens(new_emojis)

        # Figure out foundation model hidden size (fallback 768)
        try:
            hf_cfg = AutoConfig.from_pretrained(config.foundation_model.model_name)
            hidden_size = getattr(hf_cfg, "hidden_size", 768)
        except Exception:
            hidden_size = 768

        # Simple linear projection (no bias) 300 -> hidden
        in_dim = len(next(iter(self.emoji2vec_raw.values()), [])) or 300
        self.emoji_projection = nn.Linear(in_dim, hidden_size, bias=False)

        with torch.no_grad():
            for emo, vec in self.emoji2vec_raw.items():
                v = torch.tensor(vec, dtype=torch.float)
                if v.numel() != in_dim:
                    continue
                self.projected_emoji_vectors[emo] = self.emoji_projection(v).detach()

    def normalize_text(self, text):
        """Normalize tweet text.

        Steps:
        1. Replace @username -> 'người dùng'
        2. Replace links -> 'Liên kết truy cập'
        3. Lowercase
        4. Collapse repeated punctuation / symbols (e.g. '!!!!!' -> '!')
        5. Remove special symbols (keep letters, numbers, whitespace, basic punctuation and emoticons)
        """
        if not text:
            return ""

        # Replace usernames (@word)
        text = re.sub(r"@[A-Za-z0-9_]+", "người dùng", text)

        # Replace links (http/https/www + typical t.co shorteners)
        text = re.sub(r"https?://\S+", "Liên kết truy cập", text)
        text = re.sub(r"www\.\S+", "Liên kết truy cập", text)

        # Lowercase
        text = text.lower()

        # Collapse repeated punctuation / symbols (but preserve emoticons)
        text = re.sub(r"([!?.:,;\-_/\\])\1{1,}", r"\1", text)

        # Keep emoticons intact by not doing aggressive character filtering
        # Just normalize extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def encode_emoji(self, text):
        """Replace emoticons with their descriptions from the emoticon dataset.

        Each emoticon is replaced by a space + description + space to ensure token separation.
        Unknown emoticons (not in dataset) are left as-is.
        Works with multi-character emoticons like :), :D, (╯°□°)╯, etc.
        """
        if not text:
            return ""

        # Sort emoticons by length (longest first) to avoid partial matches
        emoticons_by_length = sorted(self.emoji_map.keys(), key=len, reverse=True)
        
        # Replace each emoticon with its description
        for emoticon in emoticons_by_length:
            if emoticon in text:
                desc = self.emoji_map[emoticon]
                # Use space separation to ensure tokenizer treats description as separate tokens
                text = text.replace(emoticon, f" {desc} ")
        
        # Collapse spaces created by replacements
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_text(self, text):
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=config.training.max_length)
        return encoded['input_ids'], encoded['attention_mask']

    def __call__(self, input):
        from tqdm import tqdm
        
        ids = []
        print(f"🔄 Processing {len(input)} samples with emoticon encoding...")
        
        for tweet in tqdm(input, desc="Processing tweets", unit="tweet"):
            raw_text = self.normalize_text(tweet)
            raw_text = self.encode_emoji(raw_text) if self.method == EMOJI2DESCRIPTION_METHOD else raw_text
            ids.append(self.tokenize_text(raw_text))

        return ids