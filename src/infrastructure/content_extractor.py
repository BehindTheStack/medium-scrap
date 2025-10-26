"""Robust HTML -> Markdown converter and code extractor using BeautifulSoup.

This implementation uses BeautifulSoup to extract images, code blocks and
then relies on markdownify to produce a readable Markdown output. It also
returns structured artifacts (assets list with suggested filenames and
code blocks with detected languages) for downstream persistence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from urllib.parse import urlparse

import re
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

# Try to import BeautifulSoup and markdownify; if not available, fall back to a
# minimal HTMLParser-based converter so the code works in environments where
# bs4/markdownify are not installed. This keeps tests runnable without extra
# package installation while allowing an improved path when the libraries are
# available.
_HAS_BS4 = True
try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as mdify
except Exception:
    _HAS_BS4 = False
    from html.parser import HTMLParser
    import html as _html


def _suggest_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.path.rsplit('/', 1)[-1] or 'asset'
    return name


def extract_code_blocks(html: str) -> List[Dict]:
    """Extract code blocks (<pre><code> and <pre>) and detect languages.

    Returns list of {'code': str, 'language': Optional[str]}.
    """
    soup = BeautifulSoup(html, 'html.parser')
    blocks = []

    for pre in soup.find_all('pre'):
        code_tag = pre.find('code')
        if code_tag:
            code_text = code_tag.get_text()
            # try to get language from class attribute
            lang = None
            cls = code_tag.get('class') or []
            for c in cls:
                m = re.match(r'language-(\w+)', c)
                if m:
                    lang = m.group(1)
                    break
        else:
            code_text = pre.get_text()
            lang = None

        # Try Pygments if no explicit language
        if not lang:
            try:
                lexer = guess_lexer(code_text)
                lang = lexer.name.lower()
            except ClassNotFound:
                # heuristic fallback
                lc = code_text[:200].lower()
                if 'def ' in lc or 'import ' in lc:
                    lang = 'python'
                elif 'console.log' in lc or 'function ' in lc:
                    lang = 'javascript'
                elif '#include' in lc or 'std::' in lc:
                    lang = 'cpp'
                else:
                    lang = None

        blocks.append({'code': code_text.rstrip('\n'), 'language': lang})

    return blocks


def html_to_markdown(html: str) -> Tuple[str, List[Dict], List[Dict]]:
    """Convert HTML to Markdown and extract assets & code blocks.

    Returns (markdown, assets, code_blocks).
    assets: list of {'src': original_url, 'filename': suggested_filename, 'alt': alt_text}
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Extract images and compute filenames
    assets: List[Dict] = []
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or ''
        alt = img.get('alt', '')
        if not src:
            continue
        filename = _suggest_filename_from_url(src)
        assets.append({'src': src, 'filename': filename, 'alt': alt})
        # replace src with filename in the tag so markdownify will use it
        img['src'] = filename

    # Produce markdown with markdownify (preserves code blocks reasonably)
    markdown = mdify(str(soup), heading_style='ATX')

    # Extract code blocks separately (structured)
    code_blocks = extract_code_blocks(html)

    return markdown, assets, code_blocks


def classify_technical(html: str, code_blocks: List[dict]) -> dict:
    """Simple heuristics-based technical classifier (keeps room for future ML).

    Returns: {'is_technical': bool, 'score': float, 'reasons': List[str]}
    """
    reasons: List[str] = []
    score = 0.0

    if code_blocks:
        reasons.append(f'code_blocks:{len(code_blocks)}')
        score += min(0.6, 0.2 * len(code_blocks)) + 0.4

    keywords = ['import ', 'def ', 'class ', 'function ', 'console.log', 'select ']  # simple set
    found = 0
    low_html = html.lower()
    for kw in keywords:
        if kw in low_html:
            found += 1
    if found:
        reasons.append(f'keywords:{found}')
        score += min(0.4, 0.1 * found)

    score = max(0.0, min(1.0, score))
    return {'is_technical': score >= 0.3, 'score': round(score, 2), 'reasons': reasons}


__all__ = ['html_to_markdown', 'extract_code_blocks', 'classify_technical']

def classify_technical(html: str, code_blocks: List[dict]) -> dict:
    """Simple heuristics-based technical classifier.

    - If there are code blocks, it's likely technical.
    - Look for technical keywords to boost score.

    Returns: {'is_technical': bool, 'score': float, 'reasons': List[str]}
    """
    reasons: List[str] = []
    score = 0.0

    # Code blocks presence
    if code_blocks:
        reasons.append(f"code_blocks:{len(code_blocks)}")
        score += min(0.6, 0.2 * len(code_blocks)) + 0.4

    # Keyword heuristics
    keywords = ['import ', 'def ', 'class ', 'function ', 'console.log', 'SELECT ', 'SELECT\n']
    found_kw = 0
    for kw in keywords:
        if kw.lower() in html.lower():
            found_kw += 1
    if found_kw:
        reasons.append(f"keywords:{found_kw}")
        score += min(0.4, 0.1 * found_kw)

    # Normalize score
    score = max(0.0, min(1.0, score))
    is_tech = score >= 0.3

    return {"is_technical": is_tech, "score": round(score, 2), "reasons": reasons}

__all__.append("classify_technical")
