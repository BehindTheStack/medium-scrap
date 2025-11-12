"""
Text cleaning and chunking helpers for ML pipelines.

This module centralizes markdown/HTML cleaning and provides an
optional chunking utility. By default it does NOT truncate content.
If chunking is required for a specific model, callers should request
it explicitly by passing a max_chars value.
"""
from typing import List, Optional
import re


def clean_markdown(text: Optional[str], preserve_structure: bool = True) -> str:
    """Clean markdown/HTML and normalize whitespace.

    - Removes images and badges, but keeps alt text.
    - Replaces markdown links with their text.
    - Strips code fences markers but preserves code content.
    - Removes raw HTML tags.
    - Normalizes whitespace and preserves paragraph breaks.

    Args:
        text: The markdown text to clean
        preserve_structure: If True, keeps more whitespace and structure for ML

    This function intentionally does NOT truncate the text. It returns
    the full cleaned content so downstream ML components can decide how
    to chunk or consume it.
    """
    if not text:
        return ""

    # Convert to str (defensive)
    text = str(text)

    # Remove image tags but keep alt text: ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)

    # Replace markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove standalone URLs
    text = re.sub(r'(?<!\]\()https?://\S+', '', text)

    # Remove fenced code markers but keep content inside
    text = re.sub(r'```[a-zA-Z0-9\-]*\n', '\n', text)
    text = text.replace('```', '\n')

    # Remove inline code ticks but keep content
    text = text.replace('`', '')

    # Remove common markdown header characters but keep header text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove excessive formatting characters (but preserve single * and _)
    text = re.sub(r'[\*\_]{2,}', ' ', text)
    
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)

    if preserve_structure:
        # Keep paragraph structure for TF-IDF and clustering
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Normalize spaces within lines
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        # More aggressive normalization
        text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()

    return text


def chunk_text(text: str, max_chars: Optional[int] = None, overlap: int = 0) -> List[str]:
    """Chunk text into pieces of max_chars with optional overlap.

    - If max_chars is None or larger than the text length, returns [text].
    - overlap is the number of characters that overlap between chunks.

    This helper is provided for models that cannot accept arbitrarily long
    contexts. It is intentionally opt-in so callers that want "no limits"
    can pass max_chars=None.
    """
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text]

    if overlap < 0:
        overlap = 0

    chunks = []
    start = 0
    step = max_chars - overlap if max_chars > overlap else max_chars
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += step

    return chunks
