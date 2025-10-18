import emoji
import re
import pandas as pd
import os
import torch
from typing import Tuple, List
from underthesea import text_normalize
from tqdm import tqdm

from viemonet.config import config
from viemonet.constant.training_constant import METHOD, FOUNDATION_MODEL_LIST
from viemonet.models.foundation_model_manager import FoundationModelManager
from viemonet.models.emotion_sentiment import EmotionSentiment


class PreprocessPipeline:
    def __init__(self, method=None, foundation_model_name=None, tokenizer=None):
        assert method in METHOD, \
            f"Method {method} not recognized. Available methods: {METHOD}"
        if tokenizer is None:
            assert foundation_model_name in FOUNDATION_MODEL_LIST, \
                f"Foundation model {foundation_model_name} not recognized. Available models: {FOUNDATION_MODEL_LIST}"

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            _, self.tokenizer, _ = FoundationModelManager().get_model(foundation_model_name)
        self.method = method
        self.remove_emotions = method == METHOD[2] or method == METHOD[3]
        self.emoticon_lexicon = EmotionSentiment().get_all_emoticons()
        # Sort emoticons by length (descending) for greedy longest-match
        self.emoticon_lexicon_sorted = sorted(self.emoticon_lexicon, key=len, reverse=True)
        
        # Load Vietnamese abbreviation dictionary
        abbrev_path = os.path.join(os.path.dirname(__file__), 'vietnamese_abbrev.csv')
        abbrev_df = pd.read_csv(abbrev_path)
        self.abbrev_dict = dict(zip(abbrev_df['abbrev'], abbrev_df['formal']))
        
    def split_comment_emotion(self, text: str) -> Tuple[str, List[str]]:
        """Separate plain comment from emotions (emojis and emoticons)."""
        if not text or not isinstance(text, str):
            return "", []
        
        emotions = []
        char_mask = [False] * len(text)
        
        # Extract emojis first (higher priority)
        for match in emoji.emoji_list(text):
            emotions.append(match['emoji'])
            for i in range(match['match_start'], match['match_end']):
                char_mask[i] = True
        
        # Extract emoticons with longest-match greedy approach
        i = 0
        while i < len(text):
            if char_mask[i]:
                i += 1
                continue
            
            matched = False
            for emoticon in self.emoticon_lexicon_sorted:
                if i + len(emoticon) <= len(text) and text[i:i + len(emoticon)] == emoticon:
                    if not any(char_mask[i:i + len(emoticon)]):
                        # Consume repeated trailing characters (e.g., :))))) )
                        end_pos = i + len(emoticon)
                        last_char = emoticon[-1]
                        while end_pos < len(text) and text[end_pos] == last_char and not char_mask[end_pos]:
                            end_pos += 1
                        
                        emotions.append(text[i:end_pos])
                        for j in range(i, end_pos):
                            char_mask[j] = True
                        i = end_pos
                        matched = True
                        break
            
            if not matched:
                i += 1
        
        # Build plain comment by removing marked positions
        plain_comment = ''.join(char for i, char in enumerate(text) if not char_mask[i]).strip()
        return plain_comment, emotions
    
    def emo_normalize(self, emotions: List[str]) -> List[str]:
        """
        Normalize emotions (emoticons and emojis).
        
        For emoticons:
        - Normalize informal forms like :))))) -> :)
        - Normalize repeated characters like :00000 -> :o
        - Match to closest formal emoticon in lexicon
        
        For emojis:
        - Convert to Unicode codepoint format (0x...)
        
        Args:
            emotions: List of extracted emotions (emojis and emoticons)
            
        Returns:
            List of normalized emotions
        """
        if not emotions:
            return []
        
        normalized = []
        
        for emo in emotions:
            if not emo:
                continue
            
            # Check if it's an emoji (using emoji library)
            if emoji.is_emoji(emo):
                # Convert emoji to Unicode codepoint format
                unicode_codepoint = self._emoji_to_unicode(emo)
                normalized.append(unicode_codepoint)
            else:
                # It's an emoticon - normalize it
                normalized_emoticon = self._normalize_emoticon(emo)
                if normalized_emoticon:
                    normalized.append(normalized_emoticon)
        
        return normalized
    
    def _emoji_to_unicode(self, emoji_char: str) -> str:
        """
        Convert emoji to Unicode codepoint format (0x...).
        
        Args:
            emoji_char: Single emoji character (may be multi-codepoint)
            
        Returns:
            Unicode representation in format like "0x1f600" or "0x1f468_0x200d_0x1f4bb" for multi-codepoint
        """
        # Get all codepoints for the emoji (some emojis are composed of multiple codepoints)
        codepoints = [f"0x{ord(char):x}" for char in emoji_char]
        
        # Join with underscore if multiple codepoints (e.g., for combined emojis)
        return '_'.join(codepoints)
    
    def _normalize_emoticon(self, emoticon: str) -> str:
        """
        Normalize informal emoticon forms to formal versions.
        
        Strategy:
        1. First check if it's already in lexicon (formal form)
        2. If not, try to normalize repeated characters
        3. Match to closest emoticon in lexicon
        
        Examples:
        - :))))) -> :)
        - :00000 -> :o
        - :---) -> :-)
        - =)))) -> =)
        
        Args:
            emoticon: Emoticon string (possibly informal)
            
        Returns:
            Normalized emoticon or None if no match found
        """
        # If already in lexicon, return as-is
        if emoticon in self.emoticon_lexicon:
            return emoticon
        
        # Try to normalize by removing repeated characters
        # Strategy: reduce consecutive repeated characters to single or double occurrence
        normalized_candidates = []
        
        # Generate candidate 1: reduce all repeated chars to single
        candidate1 = self._reduce_repeated_chars(emoticon, keep=1)
        if candidate1 in self.emoticon_lexicon:
            return candidate1
        normalized_candidates.append(candidate1)
        
        # Generate candidate 2: reduce all repeated chars to double
        candidate2 = self._reduce_repeated_chars(emoticon, keep=2)
        if candidate2 in self.emoticon_lexicon:
            return candidate2
        normalized_candidates.append(candidate2)
        
        # Generate candidate 3: try common character substitutions
        # Some informal emoticons use repeated separators like :--- instead of :-
        candidate3 = re.sub(r'(-{2,})', '-', emoticon)  # Multiple dashes to single dash
        if candidate3 in self.emoticon_lexicon:
            return candidate3
        
        candidate4 = re.sub(r'(={2,})', '=', emoticon)  # Multiple equals to single equal
        if candidate4 in self.emoticon_lexicon:
            return candidate4
        
        # Try reducing the candidate3 and candidate4
        candidate5 = self._reduce_repeated_chars(candidate3, keep=1)
        if candidate5 in self.emoticon_lexicon:
            return candidate5
        
        candidate6 = self._reduce_repeated_chars(candidate4, keep=1)
        if candidate6 in self.emoticon_lexicon:
            return candidate6
        
        # If no exact match found, try fuzzy matching
        # Find the closest emoticon by edit distance (considering only similar structure)
        best_match = self._fuzzy_match_emoticon(emoticon)
        if best_match:
            return best_match
        
        # If still no match, return the most reduced form
        return candidate1
    
    def _reduce_repeated_chars(self, text: str, keep: int = 1) -> str:
        """
        Reduce consecutive repeated characters.
        
        Args:
            text: Input text
            keep: Number of consecutive chars to keep (1 or 2)
            
        Returns:
            Text with repeated chars reduced
        """
        if keep == 1:
            # Reduce any sequence of 3+ identical chars to single char
            result = re.sub(r'(.)\1{2,}', r'\1', text)
        else:
            # Reduce any sequence of 3+ identical chars to double char
            result = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return result
    
    def _fuzzy_match_emoticon(self, emoticon: str) -> str:
        """
        Find the most similar emoticon in lexicon using fuzzy matching.
        
        Strategy:
        - Compare structure (same length, similar characters)
        - Prioritize emoticons with same starting character
        - Use character frequency similarity
        
        Args:
            emoticon: Emoticon to match
            
        Returns:
            Best matching emoticon from lexicon or None
        """
        if len(emoticon) == 0:
            return None
        
        # Filter lexicon by similar length (±2 characters)
        similar_length = [
            emo for emo in self.emoticon_lexicon 
            if abs(len(emo) - len(emoticon)) <= 2
        ]
        
        if not similar_length:
            return None
        
        # Further filter by same starting character
        same_start = [
            emo for emo in similar_length 
            if emo[0] == emoticon[0]
        ]
        
        if same_start:
            # If we have emoticons with same start, pick the closest length
            best_match = min(same_start, key=lambda x: abs(len(x) - len(emoticon)))
            return best_match
        
        # Otherwise, pick from similar length
        best_match = min(similar_length, key=lambda x: abs(len(x) - len(emoticon)))
        return best_match
    
    def _remove_emotions(self, text: str) -> str:
        """
        Remove all emojis and emoticons from text.
        
        Args:
            text: Input text with emotions
            
        Returns:
            Text with all emotions removed
        """
        if not text or not isinstance(text, str):
            return ""
        
        char_mask = [False] * len(text)
        
        # Mark emojis for removal
        for match in emoji.emoji_list(text):
            for i in range(match['match_start'], match['match_end']):
                char_mask[i] = True
        
        # Mark emoticons for removal with longest-match greedy approach
        i = 0
        while i < len(text):
            if char_mask[i]:
                i += 1
                continue
            
            matched = False
            for emoticon in self.emoticon_lexicon_sorted:
                if i + len(emoticon) <= len(text) and text[i:i + len(emoticon)] == emoticon:
                    if not any(char_mask[i:i + len(emoticon)]):
                        # Consume repeated trailing characters (e.g., :))))) )
                        end_pos = i + len(emoticon)
                        last_char = emoticon[-1]
                        while end_pos < len(text) and text[end_pos] == last_char and not char_mask[end_pos]:
                            end_pos += 1
                        
                        for j in range(i, end_pos):
                            char_mask[j] = True
                        i = end_pos
                        matched = True
                        break
            
            if not matched:
                i += 1
        
        # Build text by removing marked positions
        cleaned_text = ''.join(char for i, char in enumerate(text) if not char_mask[i]).strip()
        # Clean up multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text
    
    def common_preprocess(self, comment: str):
        """
        Normalize Vietnamese text using underthesea + custom abbreviation dictionary.
        
        Process:
        1. (Optional) Remove emojis and emoticons if remove_emotions=True
        2. Remove units from numbers (40k → 40, 3 ngày → 3)
        3. Apply underthesea text_normalize for general normalization
        4. Apply custom abbreviation dictionary for missed cases
        5. Tokenize the comment text
        
        Args:
            comment: Input comment text
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if not comment or not isinstance(comment, str):
            # Return empty encoding for empty comments
            max_length = config.training_setting.max_length
            return {
                'input_ids': [0] * max_length,
                'attention_mask': [0] * max_length
            }
        
        # Step 0: Remove emotions if requested
        if self.remove_emotions:
            comment = self._remove_emotions(comment)
        
        # Step 1: Remove units from numbers (keep only the number)
        normalized = self._remove_number_units(comment)
        
        # Step 2: Apply underthesea normalization
        normalized = text_normalize(normalized)
        
        # Step 3: Apply custom abbreviation dictionary
        words = normalized.split()
        processed_words = []
        
        for word in words:
            lower_word = word.lower()
            
            # Check if it's a standalone abbreviation
            if lower_word in self.abbrev_dict:
                # Preserve original casing pattern
                if word.isupper():
                    replacement = self.abbrev_dict[lower_word].upper()
                elif word[0].isupper() if word else False:
                    replacement = self.abbrev_dict[lower_word].capitalize()
                else:
                    replacement = self.abbrev_dict[lower_word]
                processed_words.append(replacement)
            else:
                # Check if word has punctuation at the end
                stripped = lower_word.rstrip('.,!?;:')
                suffix = lower_word[len(stripped):]
                
                if stripped in self.abbrev_dict:
                    replacement = self.abbrev_dict[stripped]
                    if word[0].isupper() if word else False:
                        replacement = replacement.capitalize()
                    processed_words.append(replacement + suffix)
                else:
                    processed_words.append(word)
        
        normalized = ' '.join(processed_words)
        
        # Step 4: Final cleanup
        normalized = normalized.strip()
        
        # Step 5: Tokenize the comment (returns dict with input_ids and attention_mask)
        encoded = self._tokenize_comment(normalized)
        
        return encoded
    
    def _tokenize_comment(self, text: str):
        """
        Tokenize the comment text using the foundation model's tokenizer with truncation and padding.
        
        Args:
            text: Input normalized text
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' lists
        """
        if not text:
            max_length = config.training_setting.max_length
            return {
                'input_ids': [1] * max_length,
                'attention_mask': [0] * max_length,
                'token_type_ids': [0] * max_length
            }
        
        # Encode with truncation and padding, then convert back to tokens
        max_length = config.training_setting.max_length
        
        # Use encode_plus to get both input_ids and handle truncation/padding
        encoded = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        return encoded

    def _remove_number_units(self, text: str) -> str:
        """
        Remove units from numbers, keeping only the numeric value.
        
        Examples:
            40k → 40
            200 triệu → 200
            3 ngày → 3
            5 sao → 5
            30 phút → 30
        
        Args:
            text: Input text with numbers and units
            
        Returns:
            Text with units removed
        """
        import re
        
        # Pattern: number + optional space + unit
        # Remove currency units
        text = re.sub(r'\b(\d+[\.,]?\d*)\s?(k|triệu|tr|nghìn|ngàn|đồng|VND|vnđ|\$)\b', 
                     r'\1', text, flags=re.IGNORECASE)
        
        # Remove time units
        text = re.sub(r'\b(\d+)\s?(phút|giờ|ngày|tuần|tháng|năm)\b', 
                     r'\1', text, flags=re.IGNORECASE)
        
        # Remove quantity/measurement units
        text = re.sub(r'\b(\d+)\s?(miếng|cái|chiếc|bát|đĩa|ly|chai|hộp|kg|g|ml|l|con|người)\b', 
                     r'\1', text, flags=re.IGNORECASE)
        
        # Remove rating units
        text = re.sub(r'\b(\d+[\.,]?\d*)\s?(sao|stars?)\b', 
                     r'\1', text, flags=re.IGNORECASE)
        
        # Remove percentage sign
        text = re.sub(r'\b(\d+[\.,]?\d*)\s?%', r'\1', text)
        
        return text
        
    def __call__(self, raw_data):
        # raw_data = raw_data[:3]
        processed_data = []
        for row in tqdm(raw_data, desc="Processing comments", unit="comment"):
            if self.method == METHOD[0]:
                comment, emo = self.split_comment_emotion(row)
                emo = self.emo_normalize(emotions=emo)
                comment = self.common_preprocess(comment)
                processed_data.append((comment, emo))
            else:
                comment = self.common_preprocess(row)
                processed_data.append((comment, []))

        input_ids = []
        attention_mask = []
        emotions = []
        
        for comment, emo in processed_data:
            input_ids.append(comment['input_ids'])
            attention_mask.append(comment['attention_mask'])
            emotions.append(emo)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'emotions': emotions
        }

# if __name__ == "__main__":
#     # Example usage
#     pipeline = PreprocessPipeline(method='emoticon', foundation_model_name='phobert')
#     sample_text = [
#         "dm đéo hiểu sao suất 40k mà có mỗi miếng thịt :)))), trôn trôn 🥹",
#         "Bài hát này thực sự là một kiệt tác của Sơn Tùng từ trước tới giờ 😏, ai chê được bài này t cũng lạy luôn á :))))))"    
#     ]
#     pairs = pipeline(sample_text)
    
#     for comment, emotions in pairs:
#         print("Comment:", comment)
#         print("Emotions:", emotions)