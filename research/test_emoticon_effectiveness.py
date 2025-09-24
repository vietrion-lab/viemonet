#!/usr/bin/env python3
"""
Test Vietnamese Emoticon Processing Effectiveness
Kiểm tra độ hiệu quả của việc thay thế emoticon bằng description
"""

import sys
import os
sys.path.append('/home/hqvjet/Projects/kaomoji-intergated-sentiment-analysis/research')

from emotiment.dataset import tweet_dataset
from emotiment.config import config
from emotiment.preprocess.preprocess_pipeline import PreprocessPipeline
from emotiment.constant.training_constant import EMOJI2DESCRIPTION_METHOD
import random
from tqdm import tqdm


def create_test_cases():
    """Tạo test cases từ dataset có sẵn"""
    
    print("🔍 Loading tweet dataset for test cases...")
    dataset = tweet_dataset(config)
    
    # Lấy random 20 samples
    total_samples = len(dataset)
    random.seed(42)  # For reproducibility
    indices = random.sample(range(total_samples), min(20, total_samples))
    
    test_cases = []
    for idx in indices:
        sample = dataset[idx]
        if 'tweet_content' in sample:
            content = sample['tweet_content']
        elif 'content' in sample:
            content = sample['content']
        else:
            continue
            
        sentiment = sample.get('sentiment_label', 'unknown')
        test_cases.append({
            'original_text': content,
            'sentiment': sentiment,
            'index': idx
        })
    
    return test_cases


def test_emoticon_processing():
    """Test emoticon processing với tqdm progress"""
    
    print("🎯 Vietnamese Emoticon Processing Effectiveness Test")
    print("=" * 60)
    
    # Initialize preprocessing pipeline
    print("🔄 Initializing preprocessing pipeline...")
    pipeline = PreprocessPipeline(method=EMOJI2DESCRIPTION_METHOD)
    print(f"📊 Loaded {len(pipeline.emoji_map):,} emoticons")
    
    # Create test cases
    test_cases = create_test_cases()
    print(f"📋 Created {len(test_cases)} test cases")
    
    if len(test_cases) == 0:
        print("❌ No test cases found!")
        return
    
    print("\n🧪 TESTING EMOTICON PROCESSING:")
    print("=" * 60)
    
    results = []
    
    # Process with tqdm progress bar
    for i, test_case in enumerate(tqdm(test_cases, desc="Processing samples", unit="sample")):
        original = test_case['original_text']
        sentiment = test_case['sentiment']
        
        # Step 1: Normalize text
        normalized = pipeline.normalize_text(original)
        
        # Step 2: Encode emoticons (replace with descriptions)
        encoded = pipeline.encode_emoji(normalized)
        
        # Step 3: Tokenize
        input_ids, attention_mask = pipeline.tokenize_text(encoded)
        
        # Check if any emoticons were replaced
        emoticons_found = []
        for emoticon in pipeline.emoji_map.keys():
            if emoticon in original:
                emoticons_found.append(emoticon)
        
        # Count changes
        has_emoticons = len(emoticons_found) > 0
        text_changed = normalized != encoded
        
        result = {
            'index': i + 1,
            'original_length': len(original),
            'processed_length': len(encoded),
            'emoticons_found': emoticons_found,
            'has_emoticons': has_emoticons,
            'text_changed': text_changed,
            'sentiment': sentiment,
            'original': original,
            'normalized': normalized,
            'encoded': encoded,
            'token_count': len([t for t in input_ids if t != pipeline.tokenizer.pad_token_id])
        }
        results.append(result)
    
    # Print results
    print("\n📊 PROCESSING RESULTS:")
    print("=" * 60)
    
    emoticon_samples = 0
    no_emoticon_samples = 0
    total_emoticons_found = 0
    
    for result in results:
        print(f"\n🔍 Sample {result['index']}:")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Original ({result['original_length']} chars): {result['original'][:100]}{'...' if len(result['original']) > 100 else ''}")
        print(f"   Normalized: {result['normalized'][:100]}{'...' if len(result['normalized']) > 100 else ''}")
        print(f"   Encoded ({result['processed_length']} chars): {result['encoded'][:100]}{'...' if len(result['encoded']) > 100 else ''}")
        print(f"   Tokens used: {result['token_count']}/160")
        
        if result['has_emoticons']:
            emoticon_samples += 1
            total_emoticons_found += len(result['emoticons_found'])
            print(f"   ✅ Emoticons found ({len(result['emoticons_found'])}): {result['emoticons_found']}")
            print(f"   📝 Text changed: {result['text_changed']}")
        else:
            no_emoticon_samples += 1
            print(f"   ❌ No emoticons detected")
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print("=" * 40)
    print(f"📊 Total samples: {len(results)}")
    print(f"✅ Samples with emoticons: {emoticon_samples} ({(emoticon_samples/len(results)*100):.1f}%)")
    print(f"❌ Samples without emoticons: {no_emoticon_samples} ({(no_emoticon_samples/len(results)*100):.1f}%)")
    print(f"🎯 Total emoticons found: {total_emoticons_found}")
    print(f"📊 Average emoticons per sample: {(total_emoticons_found/len(results)):.2f}")
    
    if emoticon_samples > 0:
        print(f"📊 Average emoticons per emoticon-containing sample: {(total_emoticons_found/emoticon_samples):.2f}")
    
    # Show effectiveness
    changed_samples = sum(1 for r in results if r['text_changed'])
    print(f"🔄 Samples with text changes: {changed_samples} ({(changed_samples/len(results)*100):.1f}%)")
    
    # Length analysis
    avg_original_length = sum(r['original_length'] for r in results) / len(results)
    avg_processed_length = sum(r['processed_length'] for r in results) / len(results)
    
    print(f"📏 Average original length: {avg_original_length:.1f} chars")
    print(f"📏 Average processed length: {avg_processed_length:.1f} chars")
    print(f"📊 Length change: {((avg_processed_length - avg_original_length) / avg_original_length * 100):+.1f}%")
    
    # Token usage analysis
    avg_tokens = sum(r['token_count'] for r in results) / len(results)
    print(f"🎯 Average tokens used: {avg_tokens:.1f}/160 ({(avg_tokens/160*100):.1f}%)")
    
    return results


def main():
    """Main function"""
    try:
        results = test_emoticon_processing()
        
        print("\n🎉 EFFECTIVENESS ASSESSMENT:")
        print("=" * 40)
        
        emoticon_samples = sum(1 for r in results if r['has_emoticons'])
        total_samples = len(results)
        
        if emoticon_samples > 0:
            effectiveness = (emoticon_samples / total_samples) * 100
            print(f"✅ Emoticon processing is working!")
            print(f"🎯 Effectiveness: {effectiveness:.1f}% of samples contain emoticons")
            print(f"📊 This preprocessing step will benefit {emoticon_samples}/{total_samples} samples")
            
            if effectiveness >= 50:
                print("🚀 HIGH EFFECTIVENESS - Many samples benefit from emoticon processing")
            elif effectiveness >= 20:
                print("📈 MODERATE EFFECTIVENESS - Some samples benefit from emoticon processing") 
            else:
                print("⚠️  LOW EFFECTIVENESS - Few samples contain emoticons")
        else:
            print("❌ No emoticons found in test samples")
            print("⚠️  Emoticon processing may not be very effective on this dataset")
        
        print(f"\n💡 Recommendation: {'Keep' if emoticon_samples > 0 else 'Consider reviewing'} emoticon processing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
