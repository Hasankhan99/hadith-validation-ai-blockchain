"""
Arabic Text Preprocessing and Normalization Module for Hadith Validation Pipeline.
"""

import re
import unicodedata

def remove_diacritics(text: str) -> str:
    """Removes Arabic vowel marks (tashkeel / harakat)."""
    tashkeel_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(tashkeel_pattern, '', text)

def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic letter variations:
    - Normalizes Alef forms (أ, إ, آ -> ا)
    - Normalizes Yeh (ى -> ي)
    - Normalizes Teh Marbuta (ة -> ه)
    - Removes Tatweel / Kashida (ـ)
    """
    if not isinstance(text, str):
        return ""
        
    text = remove_diacritics(text)
    
    # Remove Tatweel
    text = re.sub(r'ـ', '', text)
    
    # Normalize Alefs
    text = re.sub(r'[إأآ]', 'ا', text)
    
    # Normalize Yeh
    text = re.sub(r'ى', 'ي', text)
    
    # Normalize Teh Marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # Remove non-Arabic punctuation / digits if needed, keep basic spacing
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # Compress multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    sample_text = "حَدَّثَنَا مُحَمَّدُ بْنُ بَشَّارٍ، قَالَ: حَدَّثَنَا غُنْدَرٌ، قَالَ: حَدَّثَنَا شُعْبَةُ"
    print("Original:", sample_text)
    print("Normalized:", normalize_arabic(sample_text))
