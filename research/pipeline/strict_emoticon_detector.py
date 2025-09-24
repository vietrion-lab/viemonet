"""
Strict Emoticon/Kaomoji Detection and Filtering
Only text-based emoticons and kaomoji, no emoji images
"""

import pandas as pd
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StrictEmoticonDetector:
    def __init__(self):
        """Initialize strict emoticon detection patterns"""
        
        # Western emoticons - text only patterns
        self.western_emoticon_patterns = [
            # Basic smileys
            r':\)+',      # :), :)), :)))
            r':\(+',      # :(, :((, :(((
            r':\|',       # :|
            r':-\)',      # :-)
            r':-\(',      # :-(
            r':-\|',      # :-|
            
            # Extended expressions
            r':[DdPpOo0]+',    # :D, :d, :P, :p, :O, :o, :0
            r':[SsXx]+',       # :S, :s, :X, :x
            r':[3]+',          # :3
            r':\*+',           # :*
            # Remove problematic pattern: r':[\/\\]+' (matches :// in URLs)
            r':/(?!/)',        # :/ but not ://
            r':\\\\',          # :\
            
            # With semicolon
            r';\)+',      # ;), ;))
            r';\(',       # ;(
            r';[DdPpOo0]+',    # ;D, ;P, ;O
            r';\*',       # ;*
            r';[\/\\]',   # ;/, ;\
            
            # With equals
            r'=\)+',      # =), =))
            r'=\(',       # =(
            r'=[DdPpOo0]+',    # =D, =P, =O
            r'=\*',       # =*
            r'=[\/\\]',   # =/, =\
            
            # With caret
            r'\^\)+',     # ^), ^))
            r'\^\(',      # ^(
            r'\^[DdPpOo0]+',   # ^D, ^P, ^O
            r'\^\*',      # ^*
            r'\^[\/\\]',  # ^/, ^\
            
            # X expressions (with word boundaries)
            r'\b[Xx][DdPpOo0]+\b', # xD, XD, xP, XP
            r'\b[Xx]\)+',          # x), X)
            r'\b[Xx]\(',           # x(, X(
            
            # Remove complex patterns that might cause false positives
            # r':-?[DdPpOo0\)\(\|\/\\SsXx3\*]+',  # Too broad
            # r';-?[DdPpOo0\)\(\|\/\\SsXx3\*]+',  # Too broad
            # r'=-?[DdPpOo0\)\(\|\/\\SsXx3\*]+',  # Too broad
            # r'\^-?[DdPpOo0\)\(\|\/\\SsXx3\*]+', # Too broad
            
            # Underline expressions (more specific)
            r'[>\<]+[_\.\-]+[>\<]+',    # >_<, <_>, >.<, <.>
            r'[oO]+[_\.\-]+[oO]+',      # o_o, O_O, o.o (exclude 0 to avoid numbers)
            r'[TtYy]+[_\.\-]+[TtYy]+',  # T_T, T.T, Y_Y
            r'[@]+[_\.\-]+[@]+',        # @_@, @.@
            r'[\+]+[_\.\-]+[\+]+',      # +_+, +.+
            r'[=]+[_\.\-]+[=]+',        # =_=, =.=
            r'[\^]+[_\.\-]+[\^]+',      # ^_^, ^.^
            # Remove problematic pattern: r'[-]+[_\.\-]+[-]+' (too broad, matches -----)
            r'-_-',                     # Only specific -_-
            r'-\.-',                    # Only specific -.-
            
            # Special Vietnamese style
            r':\)\)+',    # :)), :))), :))))
            r'=\)\)+',    # =)), =))), =))))
            r';\)\)+',    # ;)), ;))), ;))))
            
            # v expressions (Vietnamese style)
            r':[vV]+',    # :v, :V, :vv
            r';[vV]+',    # ;v, ;V
            r'=[vV]+',    # =v, =V
            
            # uwu, owo expressions - standalone only
            r'\b[uU][wW][uU]\b',  # uwu, UwU (word boundaries)
            r'\b[oO][wW][oO]\b',  # owo, OwO (word boundaries)
            r'[uU][\._\-][uU]', # u_u, u.u, u-u
            r'[oO][\._\-][oO]', # o_o, o.o, o-o (already covered above but included for clarity)
        ]
        
        # Japanese kaomoji characters - only specific Unicode chars used in kaomoji
        self.kaomoji_chars = [
            'ツ', 'シ', 'ッ',  # Katakana
            'ಠ', 'ಡ', 'ಥ', 'ಯ',  # Kannada  
            '◕', '◔', '●', '◯', '○', '◎',  # Circles
            'ಠ', 'ಡ', 'ಥ', 'ಯ', 'ಸ',  # Kannada expressions
            'ლ', 'ノ', 'へ', 'ω', 'ε', 'θ',  # Various symbols
            '◖', '◗', '◘', '◙', '⊙', '◊',  # Geometric
            '｡', '･', '°', '˘', '‿', '⌒',  # Punctuation
            '∩', '∪', '⊂', '⊃', '⌐', '¬',  # Math symbols used in kaomoji
        ]
        
        # Kaomoji patterns - combinations that form faces
        self.kaomoji_patterns = [
            r'[ಠಡ◕◔●◯○◎][_\-\.]?[ಠಡ◕◔●◯○◎]',  # Eyes with optional middle
            r'\([ಠಡ◕◔●◯○◎][_\-\.][ಠಡ◕◔●◯○◎]\)',  # Eyes in parentheses
            r'[ಠಡ◕◔●◯○◎][益ᴗ▽ω∀‿⌒][ಠಡ◕◔●◯○◎]',  # Eyes with mouth
            r'[ツシッ]+',  # Japanese characters
            r'ლ\([ಠಡ◕◔●◯○◎][_\-\.][ಠಡ◕◔●◯○◎]ლ\)',  # Georgian with eyes
            r'\([ಠಡ◕◔●◯○◎][益ᴗ▽ω∀‿⌒][ಠಡ◕◔●◯○◎]\)',  # Complex faces
        ]
    
    def contains_emoji_images(self, text: str) -> bool:
        """Check if text contains emoji images (to be filtered out)"""
        if not isinstance(text, str):
            return False
        
        # Unicode ranges for emoji images (to exclude)
        emoji_image_patterns = [
            r'[\U0001F600-\U0001F64F]',  # Emoticons
            r'[\U0001F300-\U0001F5FF]',  # Misc Symbols and Pictographs
            r'[\U0001F680-\U0001F6FF]',  # Transport and Map
            r'[\U0001F1E0-\U0001F1FF]',  # Regional indicators (flags)
            r'[\U00002600-\U000027BF]',  # Miscellaneous symbols
            r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols and Pictographs
            r'[\U0001FA70-\U0001FAFF]',  # Symbols and Pictographs Extended-A
        ]
        
        for pattern in emoji_image_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def contains_western_emoticon(self, text: str) -> bool:
        """Check if text contains Western-style text emoticons"""
        if not isinstance(text, str):
            return False
        
        for pattern in self.western_emoticon_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def contains_kaomoji(self, text: str) -> bool:
        """Check if text contains Japanese-style kaomoji"""
        if not isinstance(text, str):
            return False
        
        # Check for kaomoji characters
        for char in self.kaomoji_chars:
            if char in text:
                return True
        
        # Check for kaomoji patterns
        for pattern in self.kaomoji_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def is_valid_emoticon_content(self, text: str) -> Tuple[bool, str]:
        """
        Strict validation for emoticon/kaomoji content
        Returns (is_valid, emoticon_type)
        """
        if not isinstance(text, str) or len(text.strip()) < 2:
            return False, ""
        
        text = text.strip()
        
        # First, exclude emoji images
        if self.contains_emoji_images(text):
            return False, "emoji_image"
        
        # Exclude obvious false positives
        false_positive_patterns = [
            r'^(no_opinion|experience|met_expectations|Exceeded expectations|expression_support|selfie_expert)$',
            r'Procrastination level:',  # Fix: match the phrase anywhere
            r'^\d{4}-\d{2}-\d{2}',  # dates
            r'^\d+[_\-]\d+',  # IDs with underscores/dashes but no emoticons
            r'^https?://',  # URLs
            r'^www\.',  # websites
            # Remove overly broad caps pattern
            r'^personal_\w+$',  # personal_experience etc
            # Remove this overly broad pattern: r'^[A-Za-z]+\s+[A-Za-z]+\s+[A-Za-z]+$'  # Too broad
            r'^\w+\.\w+$',  # file.extension
            r'^[A-Z]{2,}$',  # ACRONYMS
        ]
        
        for pattern in false_positive_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False, "false_positive"
        
        # Check for actual emoticons/kaomoji
        has_western = self.contains_western_emoticon(text)
        has_kaomoji = self.contains_kaomoji(text)
        
        if has_kaomoji:
            return True, "kaomoji"
        elif has_western:
            return True, "western_emoticon"
        else:
            return False, "no_emoticon"
    
    def extract_emoticons_from_text(self, text: str) -> List[str]:
        """Extract all emoticons from text"""
        if not isinstance(text, str):
            return []
        
        emoticons = []
        
        # Extract Western emoticons
        for pattern in self.western_emoticon_patterns:
            matches = re.findall(pattern, text)
            emoticons.extend(matches)
        
        # Extract kaomoji characters/patterns
        for char in self.kaomoji_chars:
            if char in text:
                emoticons.append(char)
        
        for pattern in self.kaomoji_patterns:
            matches = re.findall(pattern, text)
            emoticons.extend(matches)
        
        return list(set(emoticons))  # Remove duplicates
    
    def analyze_emoticon_content(self, text: str) -> dict:
        """Comprehensive analysis of emoticon content"""
        is_valid, emo_type = self.is_valid_emoticon_content(text)
        
        if not is_valid:
            return {
                'is_valid': False,
                'type': emo_type,
                'emoticons': [],
                'count': 0
            }
        
        emoticons = self.extract_emoticons_from_text(text)
        
        return {
            'is_valid': True,
            'type': emo_type,
            'emoticons': emoticons,
            'count': len(emoticons),
            'sample_text': text[:100] + "..." if len(text) > 100 else text
        }

def test_strict_detection():
    """Test the strict emoticon detection"""
    detector = StrictEmoticonDetector()
    
    # Test cases
    test_cases = [
        # Should PASS (valid emoticons)
        ("Hôm nay trời đẹp quá :))", True, "western_emoticon"),
        ("Buồn quá :((", True, "western_emoticon"),
        ("Cười tít mắt xD", True, "western_emoticon"),
        ("Haha =))", True, "western_emoticon"),
        ("Shock @_@", True, "western_emoticon"),
        ("Cute uwu", True, "western_emoticon"),
        ("ಠ_ಠ gì vậy", True, "kaomoji"),
        ("◕_◕ dễ thương", True, "kaomoji"),
        
        # Should FAIL (false positives)
        ("Procrastination level: expert", False, "false_positive"),
        ("Dumex Fruit & Veg số 3: 380k", False, "false_positive"),
        ("2024-01-01 12:00:00", False, "false_positive"),
        ("https://facebook.com", False, "false_positive"),
        ("CHƯƠNG TRÌNH KHUYẾN MÃI", False, "false_positive"),
        ("personal_experience", False, "false_positive"),
        ("Tôi thích ăn cơm", False, "no_emoticon"),
        
        # Should FAIL (emoji images)
        ("Hôm nay vui 😊", False, "emoji_image"),
        ("Buồn quá 😢", False, "emoji_image"),
        ("Love you 💕", False, "emoji_image"),
    ]
    
    print("🧪 Testing Strict Emoticon Detection...")
    print("="*60)
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_valid, expected_type in test_cases:
        is_valid, detected_type = detector.is_valid_emoticon_content(text)
        
        result = "✅ PASS" if (is_valid == expected_valid) else "❌ FAIL"
        if is_valid == expected_valid:
            passed += 1
        
        print(f"{result} | {text[:40]:<40} | Expected: {expected_valid:>5} ({expected_type:<15}) | Got: {is_valid:>5} ({detected_type:<15})")
    
    print("="*60)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = test_strict_detection()
    if success:
        print("🎉 All tests passed! Detection logic is working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the detection logic.")
