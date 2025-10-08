import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock

from transform_data.utils import load_config, load_dataset
from transform_data.openai_core import OpenAICore


class RateLimiter:
    """Rate limiter to control API requests per minute"""
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is more than 60 seconds old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(time.time())


def process_item(item, openai, rate_limiter, idx):
    """Process a single item with rate limiting"""
    sentence = item['Sentence']
    
    try:
        rate_limiter.wait_if_needed()
        sentiment = openai(sentence)
        return {
            "sentence": sentence,
            "sentiment": sentiment,
            "index": idx
        }
    except Exception as e:
        print(f"\nError processing item {idx}: {e}")
        return {
            "sentence": sentence,
            "sentiment": None,
            "error": str(e),
            "index": idx
        }


def process_split(split_data, openai, split_name, num_workers=10, max_rpm=450):
    """Process a dataset split with multiple workers and return results in HF format"""
    results = []
    
    print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
    print(f"Using {num_workers} workers with rate limit of {max_rpm} RPM")
    
    rate_limiter = RateLimiter(max_rpm)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_item, item, openai, rate_limiter, idx): idx 
            for idx, item in enumerate(split_data)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(split_data), desc=f"Processing {split_name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x['index'])
    
    # Remove index field from final results
    for result in results:
        result.pop('index', None)
    
    return results


def save_to_json(data, output_path):
    """Save data to JSON file in proper JSON array format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} items to {output_path}")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'transform_config.yaml') 
    config = load_config(config_path)
    dataset = load_dataset(config.dataset)
    openai = OpenAICore(**config.foundation_model.model_dump())
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    # Configuration
    num_workers = 1  # Number of concurrent workers
    max_rpm = 450     # Max requests per minute (leaving margin for safety from 500 RPM limit)
    
    # Process each split
    splits = ['train', 'validation', 'test']
    
    start_time = time.time()
    
    for split_name in splits:
        split_data = dataset[split_name]
        
        results = process_split(split_data, openai, split_name, num_workers=num_workers, max_rpm=max_rpm)
        
        # Save to JSON
        output_path = os.path.join(output_dir, f'{split_name}.json')
        save_to_json(results, output_path)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ All splits processed and saved successfully!")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    total_samples = sum(len(dataset[split]) for split in splits)
    print(f"📊 Processed {total_samples} total samples")
    print(f"⚡ Average speed: {total_samples / elapsed_time:.2f} samples/second")

