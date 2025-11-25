"""
Text cleaning and chunking helpers for ML pipelines.

This module centralizes markdown/HTML cleaning and provides an
optional chunking utility. By default it does NOT truncate content.
If chunking is required for a specific model, callers should request
it explicitly by passing a max_chars value.
"""
from typing import List, Optional
import re

# Precompile regular expressions to avoid recompilation on every call
_IMG_RE = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')
_LINK_RE = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
_URL_RE = re.compile(r'(?<!\]\()https?://\S+')
_FENCED_RE = re.compile(r'```[a-zA-Z0-9\-]*\n')
_HEADER_RE = re.compile(r'^#{1,6}\s+', flags=re.MULTILINE)
_BOLD_RE = re.compile(r'[\*\_]{2,}')
_HR_RE = re.compile(r'^[-*_]{3,}$', flags=re.MULTILINE)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_HTML_ENTITY_RE = re.compile(r'&[a-z]+;', flags=re.IGNORECASE)
_PARA_RE = re.compile(r'\n\s*\n+')
_SPACE_RE = re.compile(r'[ \t]+')
_ALL_WS_RE = re.compile(r'\s+')


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
    text = _IMG_RE.sub(r'\1', text)

    # Replace markdown links [text](url) -> text
    text = _LINK_RE.sub(r'\1', text)

    # Remove standalone URLs
    text = _URL_RE.sub('', text)

    # Remove fenced code markers but keep content inside
    text = _FENCED_RE.sub('\n', text)
    text = text.replace('```', '\n')

    # Remove inline code ticks but keep content
    text = text.replace('`', '')

    # Remove common markdown header characters but keep header text
    text = _HEADER_RE.sub('', text)

    # Remove excessive formatting characters (but preserve single * and _)
    text = _BOLD_RE.sub(' ', text)
    
    # Remove horizontal rules
    text = _HR_RE.sub('', text)

    # Remove HTML tags
    text = _HTML_TAG_RE.sub('', text)
    
    # Remove HTML entities
    text = _HTML_ENTITY_RE.sub(' ', text)

    if preserve_structure:
        # Keep paragraph structure for TF-IDF and clustering
        text = _PARA_RE.sub('\n\n', text)
        # Normalize spaces within lines
        text = _SPACE_RE.sub(' ', text)
    else:
        # More aggressive normalization
        text = _ALL_WS_RE.sub(' ', text)
    
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
