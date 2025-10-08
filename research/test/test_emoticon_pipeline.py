from viemonet.preprocess.preprocess_pipeline import PreprocessPipeline
from viemonet.dataset.curated_dataset import CuratedVietnameseDataset

dataset = CuratedVietnameseDataset().load_combined_dataset()
pipeline = PreprocessPipeline(method='emoticon')
print(f"Dataset loaded with {len(dataset)} samples.")
for text in dataset.iloc[:5]['content']:
    print(f"\nOriginal:  {text}")
    print(f"Processed:  {pipeline([text], test=True)}")
    print('-' * 30)
