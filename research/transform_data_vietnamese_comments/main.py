from datasets import load_dataset, Dataset
import re


def remove_dup(dataset):
    """
    Remove duplicate rows based on Content field.
    
    Args:
        dataset: HuggingFace Dataset object
        
    Returns:
        Dataset with duplicates removed
    """
    print(f"Removing duplicates...")
    original_size = len(dataset)
    
    # Convert to pandas for easier deduplication
    df = dataset.to_pandas()
    
    # Remove duplicates based on Content column
    df_dedup = df.drop_duplicates(subset=['Content'], keep='first')
    
    # Convert back to Dataset
    dataset_dedup = Dataset.from_pandas(df_dedup, preserve_index=False)
    
    removed = original_size - len(dataset_dedup)
    print(f"  ✓ Removed {removed} duplicates ({original_size} → {len(dataset_dedup)})")
    
    return dataset_dedup


def remove_noisy(dataset):
    """
    Remove noisy data:
    - Remove "(----- SHARES -----)" pattern from Content
    - Filter out None/empty Content
    
    Args:
        dataset: HuggingFace Dataset object
        
    Returns:
        Dataset with noisy data removed
    """
    print(f"Removing noisy data...")
    original_size = len(dataset)
    
    def clean_content(example):
        content = example['Content']
        
        # Mark None or non-string as empty
        if not content or not isinstance(content, str):
            example['Content'] = ""
            return example
        
        # Remove "(----- SHARES -----)" pattern only
        content = re.sub(r'\(-+\s*SHARES?\s*-+\)', '', content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        example['Content'] = content
        return example
    
    # Apply cleaning
    dataset_cleaned = dataset.map(clean_content)
    
    # Filter out None/empty content
    def is_valid_content(example):
        content = example['Content']
        return content and len(content.strip()) > 0
    
    dataset_filtered = dataset_cleaned.filter(is_valid_content)
    
    removed = original_size - len(dataset_filtered)
    print(f"  ✓ Removed {removed} noisy entries ({original_size} → {len(dataset_filtered)})")
    
    return dataset_filtered


def normalize_sentiment_labels(dataset):
    """
    Normalize sentiment labels to numeric values:
    - "Tích cực" / "positive" / "Positive" -> 1
    - "Trung lập" / "neutral" / "Neutral" -> 0
    - "Tiêu cực" / "negative" / "Negative" -> -1
    
    Args:
        dataset: HuggingFace Dataset object
        
    Returns:
        Dataset with normalized sentiment labels
    """
    print(f"Normalizing sentiment labels...")
    
    # Count label distribution before normalization
    label_counts = {}
    for example in dataset:
        label = example.get('Sentiment', '')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Original label distribution: {label_counts}")
    
    def normalize_label(example):
        sentiment = example.get('Sentiment', '').strip()
        
        # Map to numeric values
        if sentiment in ['Tích cực', 'positive', 'Positive']:
            example['Sentiment'] = 1
        elif sentiment in ['Trung lập', 'neutral', 'Neutral']:
            example['Sentiment'] = 0
        elif sentiment in ['Tiêu cực', 'negative', 'Negative']:
            example['Sentiment'] = -1
        else:
            # Handle unknown labels - default to neutral
            print(f"  Warning: Unknown sentiment label '{sentiment}', mapping to 0 (neutral)")
            example['Sentiment'] = 0
        
        return example
    
    # Apply label normalization
    dataset_normalized = dataset.map(normalize_label)
    
    # Count label distribution after normalization
    normalized_counts = {-1: 0, 0: 0, 1: 0}
    for example in dataset_normalized:
        label = example['Sentiment']
        normalized_counts[label] = normalized_counts.get(label, 0) + 1
    
    print(f"  ✓ Normalized labels: -1 (negative): {normalized_counts[-1]}, 0 (neutral): {normalized_counts[0]}, 1 (positive): {normalized_counts[1]}")
    
    return dataset_normalized


def normalize_dataset(dataset):
    """
    Normalize dataset by removing duplicates and noisy data.
    
    Process:
    1. Remove duplicate entries (based on Content)
    2. Remove/clean noisy patterns
    3. Normalize sentiment labels to -1, 0, 1
    4. Apply to all splits (train, validation, test)
    
    Args:
        dataset: HuggingFace DatasetDict with train/validation/test splits
        
    Returns:
        Normalized DatasetDict
    """
    print("\n" + "="*80)
    print("NORMALIZING DATASET")
    print("="*80)
    
    normalized = {}
    
    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            continue
            
        print(f"\n--- Processing {split.upper()} split ---")
        print(f"Original size: {len(dataset[split])}")
        
        # Step 1: Remove duplicates
        split_data = remove_dup(dataset[split])
        
        # Step 2: Remove noisy data
        split_data = remove_noisy(split_data)
        
        # Step 3: Normalize sentiment labels
        split_data = normalize_sentiment_labels(split_data)
        
        normalized[split] = split_data
        
        print(f"Final size: {len(split_data)}")
        print(f"Reduction: {len(dataset[split]) - len(split_data)} rows ({(1 - len(split_data)/len(dataset[split]))*100:.2f}%)")
    
    print("\n" + "="*80)
    print("NORMALIZATION COMPLETE")
    print("="*80)
    
    # Convert dict back to DatasetDict
    from datasets import DatasetDict
    return DatasetDict(normalized)

if __name__ == '__main__':
    import os
    import json
    
    # Load the dataset
    dataset = load_dataset("minhtoan/vietnamese-comment-sentiment")
    
    print("\n" + "="*80)
    print("ORIGINAL DATASET")
    print("="*80)
    print(dataset)
    
    # Normalize the dataset
    normalized_dataset = normalize_dataset(dataset)
    
    print("\n" + "="*80)
    print("NORMALIZED DATASET")
    print("="*80)
    print(normalized_dataset)
    
    # Show sample cleaned data
    print("\n" + "="*80)
    print("SAMPLE CLEANED DATA (first 3 from train split)")
    print("="*80)
    for i, example in enumerate(normalized_dataset['train'].select(range(min(3, len(normalized_dataset['train']))))):
        print(f"\n[{i+1}] Content: {example['Content'][:200]}...")
        print(f"    Sentiment: {example['Sentiment']}")
        print(f"    Topic: {example.get('Topic', 'N/A')}")
    
    # Save to output directory
    output_dir = "../transform_data_vietnamese_comments/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"SAVING TO {output_dir}")
    print("="*80)
    
    for split in ['train', 'validation', 'test']:
        if split not in normalized_dataset:
            continue
        
        output_file = os.path.join(output_dir, f"{split}.json")
        
        # Convert to list of dictionaries
        data_list = []
        for example in normalized_dataset[split]:
            data_list.append({
                'Content': example['Content'],
                'Sentiment': example['Sentiment'],
                'Topic': example.get('Topic', '')
            })
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved {len(data_list)} entries to {output_file}")
    
    print("\n" + "="*80)
    print("ALL FILES SAVED SUCCESSFULLY")
    print("="*80)
