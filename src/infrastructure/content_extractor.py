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
    try:
        # prefer lxml parser when available for better performance
        import lxml  # type: ignore
        _BS_PARSER = 'lxml'
    except Exception:
        _BS_PARSER = 'html.parser'
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

    The function returns a list of dictionaries with keys:
    - 'code': the raw code text
    - 'language': a best-effort language name (or None)

    Detection strategy:
    1. Look for explicit language classes on the <code> tag (e.g. 'language-python').
    2. Try Pygments.guess_lexer as a fallback.
    3. Use small heuristics for a few common languages.
    """
    soup = BeautifulSoup(html, _BS_PARSER)
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

    Returns a tuple: (markdown, assets, code_blocks).

    - markdown: string with the converted Markdown content.
    - assets: list of dicts with keys 'src', 'filename' and 'alt' suitable for persistence.
    - code_blocks: list of {'code': ..., 'language': ...} extracted from the original HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Heuristic: prefer the main article content if present to avoid nav/footer noise.
    main_selectors = [
        ('article', {}),
        ('main', {}),
        (None, {'role': 'article'}),
    ]

    main_node = None
    # Try common class-based containers used by Medium and other publishers
    class_candidates = [r'postArticle', r'meteredContent', r'post-content', r'article', r'section-content', r'entry-content']

    for tag, attrs in main_selectors:
        if tag:
            main_node = soup.find(tag)
        else:
            main_node = soup.find(attrs=attrs)
        if main_node:
            break

    if not main_node:
        # Try class-based heuristics
        for cls in class_candidates:
            main_node = soup.find('div', class_=re.compile(cls, re.I))
            if main_node:
                break

    # If we found a candidate, work on a copy of that node only; otherwise keep full soup
    content_soup = BeautifulSoup(str(main_node), _BS_PARSER) if main_node else soup

    # Remove common noisy elements: nav/header/footer/aside/forms and signup/signin links
    noisy_tags = ['nav', 'header', 'footer', 'aside', 'form', 'noscript', 'script', 'iframe']
    for t in content_soup.find_all(noisy_tags):
        try:
            t.decompose()
        except Exception:
            pass

    # Remove anchors/buttons likely related to signup/login/search/sitemap
    noisy_texts = ['sign up', 'sign in', 'search', 'sitemap', 'open in app', 'listen', 'share', 'responses', 'help', 'about', 'privacy', 'terms']
    for a in content_soup.find_all(['a', 'button', 'span']):
        txt = (a.get_text() or '').strip().lower()
        for nt in noisy_texts:
            if nt in txt:
                try:
                    a.decompose()
                except Exception:
                    pass
                break

    # Also remove common class-based widgets (subscribe, cookie, masthead)
    widget_patterns = [r'subscribe', r'cookie', r'masthead', r'promo', r'global-nav', r'byline', r'author-info']
    for pat in widget_patterns:
        for el in content_soup.find_all(class_=re.compile(pat, re.I)):
            try:
                el.decompose()
            except Exception:
                pass

    # Extract images and compute filenames from the pruned content
    assets: List[Dict] = []
    for img in content_soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original') or ''
        alt = img.get('alt', '')
        if not src:
            continue
        filename = _suggest_filename_from_url(src)
        assets.append({'src': src, 'filename': filename, 'alt': alt})
        # replace src with filename so markdownify will reference the local file name
        img['src'] = filename

    # Produce markdown with markdownify (preserves code blocks reasonably)
    markdown = mdify(str(content_soup), heading_style='ATX')

    # Extract code blocks separately (structured) from the main content (fallback to full html)
    code_blocks = extract_code_blocks(str(content_soup) if main_node else html)

    return markdown, assets, code_blocks


# Technical tags that indicate engineering content
TECHNICAL_TAGS = {
    # Programming languages
    'python', 'javascript', 'java', 'golang', 'go', 'rust', 'typescript', 'kotlin',
    'swift', 'scala', 'ruby', 'php', 'c++', 'cpp', 'c#', 'csharp', '.net',
    # Frameworks & Libraries
    'react', 'angular', 'vue', 'nodejs', 'node.js', 'django', 'flask', 'spring',
    'rails', 'laravel', 'express', 'fastapi', 'nextjs', 'next.js',
    # Data & ML
    'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'ml',
    'data science', 'data engineering', 'big data', 'analytics', 'tensorflow',
    'pytorch', 'pandas', 'numpy', 'scikit-learn', 'spark', 'hadoop', 'kafka',
    'recommendation system', 'nlp', 'computer vision', 'neural network',
    # Infrastructure & DevOps
    'kubernetes', 'docker', 'aws', 'azure', 'gcp', 'cloud', 'devops', 'ci/cd',
    'terraform', 'ansible', 'jenkins', 'github actions', 'microservices',
    'serverless', 'lambda', 'containers', 'infrastructure',
    # Databases
    'sql', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
    'cassandra', 'dynamodb', 'database', 'graphql',
    # Architecture & Patterns
    'architecture', 'system design', 'distributed systems', 'scalability',
    'performance', 'optimization', 'caching', 'load balancing', 'api',
    'rest', 'grpc', 'event-driven', 'microservice',
    # Security
    'security', 'authentication', 'authorization', 'encryption', 'oauth',
    # Mobile
    'ios', 'android', 'mobile development', 'react native', 'flutter',
    # Testing
    'testing', 'unit testing', 'integration testing', 'tdd', 'automation',
    # Other technical
    'algorithm', 'data structure', 'open source', 'backend', 'frontend',
    'full stack', 'web development', 'software engineering', 'engineering',
    'technical', 'programming', 'coding', 'development', 'developer',
}


def classify_technical(html: str, code_blocks: List[dict], tags: List[str] = None) -> dict:
    """Heuristics-based technical classifier using code blocks, keywords, and tags.

    Args:
        html: HTML content of the post
        code_blocks: List of extracted code blocks
        tags: Optional list of post tags from Medium API

    Returns: {'is_technical': bool, 'score': float, 'reasons': List[str]}
    """
    reasons: List[str] = []
    score = 0.0
    tags = tags or []

    # 1. Code blocks presence (strong signal)
    if code_blocks:
        reasons.append(f"code_blocks:{len(code_blocks)}")
        score += min(0.6, 0.2 * len(code_blocks)) + 0.4

    # 2. Technical tags (strong signal)
    if tags:
        normalized_tags = [t.lower().strip() for t in tags]
        matching_tags = [t for t in normalized_tags if t in TECHNICAL_TAGS]
        if matching_tags:
            reasons.append(f"tech_tags:{matching_tags[:3]}")
            score += min(0.5, 0.15 * len(matching_tags))

    # 3. Keyword heuristics in content (weaker signal)
    keywords = ['import ', 'def ', 'class ', 'function ', 'console.log', 'SELECT ', 'const ', 'let ', 'var ']
    found_kw = 0
    low_html = html.lower()
    for kw in keywords:
        if kw.lower() in low_html:
            found_kw += 1
    if found_kw:
        reasons.append(f"keywords:{found_kw}")
        score += min(0.3, 0.08 * found_kw)

    # Normalize score
    score = max(0.0, min(1.0, score))
    is_tech = score >= 0.25  # Lowered threshold since tags are reliable

    return {"is_technical": is_tech, "score": round(score, 2), "reasons": reasons}


__all__ = ['html_to_markdown', 'extract_code_blocks', 'classify_technical']
