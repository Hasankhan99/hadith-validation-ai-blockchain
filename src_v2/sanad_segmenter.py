"""
Sanad and Matn Segmentation Module.
Extracts chain of narrators (Isnad) and body text (Matn) from Hadith narration.
"""

import re
from typing import Tuple, Dict

NARRATOR_KEYWORDS = [
    r'حدثنا', r'أخبرنا', r'سمعت', r'عن', r'حدثني', r'أخبرني', r'قال', r'عن أبي', r'أن'
]

def segment_isnad_matn(hadith_text: str) -> Dict[str, str]:
    """
    Segments Hadith into Isnad and Matn using key narrator transit signals.
    """
    if not isinstance(hadith_text, str) or not hadith_text.strip():
        return {"isnad": "", "matn": ""}
        
    # Common transition phrases from Isnad to Matn:
    # "أن رسول الله", "قال رسول الله", "عن النبي", "أن النبي", "أنه قال", "يقول"
    matn_transition_patterns = [
        r'(أن رسول الله\b.*)',
        r'(قال رسول الله\b.*)',
        r'(سمعت رسول الله\b.*)',
        r'(عن النبي\b.*)',
        r'(أن النبي\b.*)',
        r'(أن رسول الله صلى الله عليه وسلم\b.*)',
        r'(قال النبي\b.*)'
    ]
    
    matn_text = ""
    isnad_text = hadith_text
    
    for pattern in matn_transition_patterns:
        match = re.search(pattern, hadith_text)
        if match:
            split_idx = match.start()
            isnad_text = hadith_text[:split_idx].strip()
            matn_text = hadith_text[split_idx:].strip()
            break
            
    # Fallback heuristic if no explicit transition phrase is matched:
    if not matn_text and len(hadith_text.split()) > 15:
        # Heuristic: First ~40% is likely Isnad, remaining is Matn
        words = hadith_text.split()
        split_point = max(5, int(len(words) * 0.35))
        isnad_text = " ".join(words[:split_point])
        matn_text = " ".join(words[split_point:])
        
    return {
        "isnad": isnad_text,
        "matn": matn_text if matn_text else hadith_text
    }

if __name__ == "__main__":
    sample = "حدثنا مسدد قال حدثنا يحيى عن شعبة عن قتادة عن أنس قال قال رسول الله صلى الله عليه وسلم لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه"
    res = segment_isnad_matn(sample)
    print("Isnad:", res["isnad"])
    print("Matn:", res["matn"])
