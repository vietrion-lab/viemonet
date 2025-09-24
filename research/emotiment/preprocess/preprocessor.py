from emotiment.preprocess.preprocess_pipeline import PreprocessPipeline
from emotiment.training.decorators import log_step
from emotiment.constant.training_constant import EMOJI2VEC_METHOD, EMOJI2DESCRIPTION_METHOD

import random
from torch.utils.data import Dataset


class _ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class TrainingPreprocessor:
    def __init__(self, method):
        self.pipeline = PreprocessPipeline(method=method)
        # Expose tokenizer and projected emoji vectors (if any) for model building
        self.tokenizer = self.pipeline.tokenizer
        self.projected_emoji_vectors = getattr(self.pipeline, 'projected_emoji_vectors', {})
        # Initialize other necessary attributes and preprocessing steps here

    @log_step
    def __call__(self, tweet_data):
        # Support both 'tweet_content' and 'content' column names
        if 'tweet_content' in tweet_data.column_names:
            tweet_contents = tweet_data['tweet_content']
        elif 'content' in tweet_data.column_names:
            tweet_contents = tweet_data['content']
        else:
            raise ValueError("Dataset must have either 'tweet_content' or 'content' column")
            
        sentiment_labels = tweet_data['sentiment_label']
        
        # Handle both text and numeric sentiment labels
        if isinstance(sentiment_labels[0], str):
            # Text labels: convert to integers
            label_to_int = {'negative': 0, 'neutral': 1, 'positive': 2}
            numeric_labels = [label_to_int[label] for label in sentiment_labels]
            unique_labels = [0, 1, 2]  # negative, neutral, positive
        else:
            # Numeric labels: use as is
            numeric_labels = [int(l) for l in sentiment_labels]
            unique_labels = sorted(set(numeric_labels))
            
        label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
        ids = self.pipeline(tweet_contents)
        dataset = []

        for i in range(len(ids)):
            input_ids, attention_mask = ids[i]
            dataset.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_map[numeric_labels[i]] if isinstance(sentiment_labels[0], int) else numeric_labels[i]
            })

        n = len(dataset)
        if n == 0:
            return [], [], []
        rng = random.Random(42)
        rng.shuffle(dataset)
        train_end = int(0.7 * n)
        eval_end = train_end + int(0.1 * n)
        if eval_end == train_end and n - train_end > 1:
            eval_end += 1
        if eval_end >= n:
            eval_end = min(n, train_end + 1)
        required = {"input_ids", "attention_mask", "labels"}
        def _clean(split):
            good = []
            for s in split:
                if isinstance(s, dict) and required.issubset(s.keys()):
                    good.append(s)
            return good
        train_set = _clean(dataset[:train_end])
        eval_set = _clean(dataset[train_end:eval_end])
        test_set = _clean(dataset[eval_end:])
        return _ListDataset(train_set), _ListDataset(eval_set), _ListDataset(test_set)