#!/usr/bin/env python3
"""
Simple Emoticon Processing Test
Test ngắn gọn để kiểm tra độ hiệu quả của emoticon processing
"""

import sys
import os
import re
import random
from tqdm import tqdm

# Add path
sys.path.append('/home/hqvjet/Projects/kaomoji-intergated-sentiment-analysis/research')

# Direct imports to avoid circular dependency
from emotiment.dataset.tweet_dataset import tweet_dataset
from emotiment.dataset.emoji_dataset import emoji_dataset
from emotiment.config.train_config import config


def simple_emoticon_test():
    """Test đơn giản emoticon processing"""
    
    print("🎯 SIMPLE EMOTICON PROCESSING TEST")
    print("=" * 50)
    
    # Load datasets
    print("🔄 Loading datasets...")
    tweets = tweet_dataset(config)
    emoticons = emoji_dataset(config)
    
    print(f"📊 Tweet dataset: {len(tweets)} samples")
    print(f"🎭 Emoticon dataset: {len(emoticons)} emoticons")
    
    # Build emoticon map
    print("🔄 Building emoticon map...")
    emoticon_map = {}
    for row in tqdm(emoticons, desc="Building map", unit="emoticon"):
        code = row.get('emoticon_code', '').strip()
        desc = row.get('description', '').strip().lower()
        if code and desc:
            emoticon_map[code] = desc
    
    print(f"✅ Built emoticon map: {len(emoticon_map)} entries")
    
    # Test on random samples
    print("\n🧪 Testing on random samples...")
    random.seed(42)
    
    # Get content field name
    content_field = 'tweet_content' if 'tweet_content' in tweets.column_names else 'content'
    sentiment_field = 'sentiment_label'
    
    # Sample 20 random tweets
    indices = random.sample(range(len(tweets)), min(20, len(tweets)))
    test_samples = [tweets[i] for i in indices]
    
    results = []
    emoticon_found_count = 0
    
    print(f"\n🔍 Processing {len(test_samples)} samples:")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples, 1):
        original_text = sample[content_field]
        sentiment = sample.get(sentiment_field, 'unknown')
        
        # Find emoticons in text
        found_emoticons = []
        processed_text = original_text
        
        # Sort emoticons by length (longest first) to avoid partial matches
        sorted_emoticons = sorted(emoticon_map.keys(), key=len, reverse=True)
        
        for emoticon in sorted_emoticons:
            if emoticon in processed_text:
                found_emoticons.append(emoticon)
                description = emoticon_map[emoticon]
                processed_text = processed_text.replace(emoticon, f" {description} ")
        
        # Clean up spaces
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # Record result
        has_emoticons = len(found_emoticons) > 0
        if has_emoticons:
            emoticon_found_count += 1
        
        text_changed = original_text != processed_text
        
        result = {
            'index': i,
            'original': original_text,
            'processed': processed_text,
            'sentiment': sentiment,
            'emoticons': found_emoticons,
            'has_emoticons': has_emoticons,
            'text_changed': text_changed,
            'original_length': len(original_text),
            'processed_length': len(processed_text)
        }
        results.append(result)
        
        # Display result
        print(f"\n📝 Sample {i}:")
        print(f"   Sentiment: {sentiment}")
        print(f"   Original ({len(original_text)} chars): {original_text[:80]}{'...' if len(original_text) > 80 else ''}")
        
        if has_emoticons:
            print(f"   ✅ Found emoticons ({len(found_emoticons)}): {found_emoticons}")
            print(f"   Processed ({len(processed_text)} chars): {processed_text[:80]}{'...' if len(processed_text) > 80 else ''}")
            print(f"   📊 Length change: {len(processed_text) - len(original_text):+d} chars")
        else:
            print(f"   ❌ No emoticons found")
            print(f"   📊 Text unchanged")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print("=" * 40)
    print(f"📈 Total samples tested: {len(results)}")
    print(f"✅ Samples with emoticons: {emoticon_found_count}/{len(results)} ({(emoticon_found_count/len(results)*100):.1f}%)")
    print(f"❌ Samples without emoticons: {len(results)-emoticon_found_count}/{len(results)} ({((len(results)-emoticon_found_count)/len(results)*100):.1f}%)")
    
    # Total emoticons found
    total_emoticons = sum(len(r['emoticons']) for r in results)
    print(f"🎭 Total emoticons detected: {total_emoticons}")
    
    if emoticon_found_count > 0:
        print(f"📊 Average emoticons per positive sample: {total_emoticons/emoticon_found_count:.1f}")
    
    # Length analysis
    original_avg = sum(r['original_length'] for r in results) / len(results)
    processed_avg = sum(r['processed_length'] for r in results) / len(results)
    
    print(f"📏 Average original length: {original_avg:.1f} chars")
    print(f"📏 Average processed length: {processed_avg:.1f} chars")
    print(f"📊 Average length change: {processed_avg - original_avg:+.1f} chars ({((processed_avg - original_avg)/original_avg*100):+.1f}%)")
    
    # Effectiveness assessment
    print(f"\n🎯 EFFECTIVENESS ASSESSMENT:")
    print("=" * 40)
    
    effectiveness = (emoticon_found_count / len(results)) * 100
    
    if effectiveness >= 50:
        print(f"🚀 HIGH EFFECTIVENESS ({effectiveness:.1f}%)")
        print("   Many samples contain emoticons - processing is very beneficial!")
    elif effectiveness >= 20:
        print(f"📈 MODERATE EFFECTIVENESS ({effectiveness:.1f}%)")
        print("   Some samples contain emoticons - processing provides moderate benefit.")
    elif effectiveness >= 5:
        print(f"⚠️  LOW EFFECTIVENESS ({effectiveness:.1f}%)")
        print("   Few samples contain emoticons - processing provides limited benefit.")
    else:
        print(f"❌ VERY LOW EFFECTIVENESS ({effectiveness:.1f}%)")
        print("   Almost no samples contain emoticons - consider reviewing approach.")
    
    print(f"\n💡 RECOMMENDATION:")
    if effectiveness >= 20:
        print("   ✅ Keep emoticon processing - it's beneficial for this dataset!")
    elif effectiveness >= 5:
        print("   ⚠️  Consider keeping emoticon processing but monitor performance impact.")
    else:
        print("   ❌ Consider removing or optimizing emoticon processing.")
    
    return results, effectiveness


def main():
    try:
        results, effectiveness = simple_emoticon_test()
        
        print(f"\n🎉 TEST COMPLETED!")
        print(f"🎯 Final effectiveness score: {effectiveness:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
